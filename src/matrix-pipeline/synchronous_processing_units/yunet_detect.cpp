#include "yunet_detect.h"
#include "../utils/cuda_helper.h"

#include <fmt/ranges.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/dnn.hpp>
#include <spdlog/spdlog.h>

namespace MatrixPipeline::ProcessingUnit {

YuNetDetect::~YuNetDetect() {
  if (m_cuda_stream)
    cudaStreamDestroy(m_cuda_stream);
}

bool YuNetDetect::init(const njson &config) {
  try {
    const std::string model_path = config.value("modelPath", "");
    m_score_threshold = config.value("scoreThreshold", 0.9f);
    m_nms_threshold = config.value("nmsThreshold", 0.3f);
    m_model_input_size =
        cv::Size(config.value("inputWidth", m_model_input_size.width),
                 config.value("inputHeight", m_model_input_size.height));
    m_inference_interval = std::chrono::milliseconds(
        config.value("inferenceIntervalMs", m_inference_interval.count()));
    {
      // 1. Build Engine (Standard Boilerplate)
      auto builder = std::unique_ptr<nvinfer1::IBuilder>(
          nvinfer1::createInferBuilder(Utils::g_logger));
      uint32_t flags =
          1U << static_cast<uint32_t>(
              nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
      auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
          builder->createNetworkV2(flags)); // kEXPLICIT_BATCH
      auto parser = std::unique_ptr<nvonnxparser::IParser>(
          nvonnxparser::createParser(*network, Utils::g_logger));
      parser->parseFromFile(model_path.c_str(), 2);

      auto trt_config = std::unique_ptr<nvinfer1::IBuilderConfig>(
          builder->createBuilderConfig());
      auto plan = std::unique_ptr<nvinfer1::IHostMemory>(
          builder->buildSerializedNetwork(*network, *trt_config));
      auto runtime = std::unique_ptr<nvinfer1::IRuntime>(
          nvinfer1::createInferRuntime(Utils::g_logger));
      m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(
          runtime->deserializeCudaEngine(plan->data(), plan->size()));
      m_context = std::unique_ptr<nvinfer1::IExecutionContext>(
          m_engine->createExecutionContext());
    }

    {
      // 2. Dynamic Buffer Allocation for 13 Tensors
      for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const char *name = m_engine->getIOTensorName(i);
        nvinfer1::Dims dims = m_engine->getTensorShape(name);

        // Calculate the total number of FLOAT elements
        size_t count = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
          count *= dims.d[j];
        }

        if (m_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
          // FIX: Remove 'sizeof(float)'. Allocate 'count' number of floats.
          m_input_buffer_gpu = Utils::make_device_unique<float>(count);

          // Bind the input tensor address
          m_context->setTensorAddress(name, m_input_buffer_gpu.get());

          SPDLOG_INFO("Input Tensor: {} | Count: {} floats", name, count);
        } else {
          // FIX: Remove 'sizeof(float)'. Allocate 'count' number of floats.
          m_gpu_buffers[name] = Utils::make_device_unique<float>(count);

          // CPU buffer resize is correct (takes number of elements)
          m_cpu_buffers[name].resize(count);

          // Bind the output tensor address
          m_context->setTensorAddress(name, m_gpu_buffers[name].get());
          SPDLOG_INFO(
              "Output Tensor: {} | Count: {} floats | Dims: {}x{}x{}x{}", name,
              count, dims.d[0], dims.d[1], dims.d[2], dims.d[3]);
        }
      }
    }

    SPDLOG_INFO("IO tensors:");
    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
      auto name = m_engine->getIOTensorName(i);
      auto dims = m_engine->getTensorShape(name);
      SPDLOG_INFO("Index {}: Name='{}', Shape=[{}]", i, name,
                  fmt::join(dims.d, "x"));
    }

    if (cudaStreamCreate(&m_cuda_stream) != cudaSuccess) {
      SPDLOG_ERROR("cudaStreamCreate() failed");
      return false;
    }
    m_cv_stream = cv::cuda::StreamAccessor::wrapStream(m_cuda_stream);

    return true;
  } catch (const std::exception &e) {
    SPDLOG_ERROR("YuNet Init: {}", e.what());
    return false;
  }
}

SynchronousProcessingResult YuNetDetect::process(cv::cuda::GpuMat &frame,
                                                 PipelineContext &ctx) {
  const auto steady_now = std::chrono::steady_clock::now();
  if (steady_now - m_last_inference_at < m_inference_interval) {
    ctx.yunet = m_prev_yunet_ctx;
    return success_and_continue;
  }
  m_last_inference_at = steady_now;

  if (frame.empty() || !m_context) {
    return failure_and_continue;
  }

  // 2. Preprocessing
  // 1. GPU Preprocessing (BGR -> RGB -> Planar NCHW)
  cv::cuda::resize(frame, m_resized_gpu, m_model_input_size, 0, 0,
                   cv::INTER_LINEAR, m_cv_stream);
  cv::cuda::cvtColor(m_resized_gpu, m_rgb, cv::COLOR_BGR2RGB, 0, m_cv_stream);
  // alpha is 1.0, while in yolo_detect.cpp it is 1.0/255.0
  m_rgb.convertTo(m_normalized_gpu, CV_32FC3, 1.0, m_cv_stream);

  /*
  cv::cuda::subtract(m_normalized_gpu, cv::Scalar(127.0, 127.0, 127.0),
                     m_normalized_gpu, cv::noArray(), -1, m_cv_stream);*/

  std::vector<cv::cuda::GpuMat> channels;
  {
    // 1. Split the 3-channel image into 3 separate planes
    std::vector<cv::cuda::GpuMat> debug_channels;
    cv::cuda::split(m_rgb, debug_channels, m_cv_stream);

    // 2. Now check just the first channel (Blue or Red)
    double minVal, maxVal;
    // Sync to ensure the split is finished before reading
    cudaStreamSynchronize(m_cuda_stream);
    cv::cuda::minMax(debug_channels[0], &minVal, &maxVal);

    SPDLOG_INFO("Input Frame Channel 0 -> Min: {:.2f}, Max: {:.2f}", minVal,
                maxVal);
  }
  cv::cuda::split(m_normalized_gpu, channels, m_cv_stream);
  // 3. Check one channel AFTER the split
  // double minV, maxV;
  // cv::cuda::minMax(channels[0], &minV, &maxV);

  // SPDLOG_INFO("Channel 0 Status -> Min: {:.2f}, Max: {:.2f}", minV, maxV);
  //  m_input_buffer_gpu is a flat float array of size 3 * W * H

  for (int i = 0; i < 3; ++i) {
    float *dest_ptr =
        m_input_buffer_gpu.get() + (i * m_model_input_size.area());
    cudaMemcpy2DAsync(dest_ptr, // Destination
                      m_model_input_size.width *
                          sizeof(float), // Dest Pitch (No padding)
                      channels[i].data,  // Source (One plane from split)
                      channels[i].step,  // Source Pitch (OpenCV padding)
                      m_model_input_size.width * sizeof(float), // Width to copy
                      m_model_input_size.height, // Height to copy
                      cudaMemcpyDeviceToDevice, m_cuda_stream);
  }
  cudaStreamSynchronize(m_cuda_stream);
  {
    // Check the center of the image (320, 320)
    size_t center_offset = (320 * 640 + 320); // Row 320, Col 320
    float center_val;

    // Sync before reading to be 100% sure
    cudaStreamSynchronize(m_cuda_stream);
    cudaMemcpy(&center_val, m_input_buffer_gpu.get() + center_offset,
               sizeof(float), cudaMemcpyDeviceToHost);

    SPDLOG_INFO("Center Float Value: {:.4f}", center_val);
  }

  for (auto &[name, cpu_buf] : m_cpu_buffers) {
    std::fill(cpu_buf.begin(), cpu_buf.end(), -100.0f);
  }
  // 2. Inference & Async Download
  m_context->enqueueV3(m_cuda_stream);
  for (auto &[name, cpu_buf] : m_cpu_buffers) {
    cudaMemcpyAsync(cpu_buf.data(), m_gpu_buffers[name].get(),
                    cpu_buf.size() * sizeof(float), cudaMemcpyDeviceToHost,
                    m_cuda_stream);
  }
  cudaStreamSynchronize(m_cuda_stream);

  // 3. Multi-Head Decoding
  decode_heads(frame, ctx);

  m_prev_yunet_ctx = ctx.yunet;
  return success_and_continue;
}

void YuNetDetect::decode_heads(const cv::cuda::GpuMat &frame,
                               PipelineContext &ctx) {
  std::vector<FaceDetection> candidates;
  std::vector<cv::Rect> bboxes;
  std::vector<float> scores;

  float sx = (float)frame.cols / m_model_input_size.width;
  float sy = (float)frame.rows / m_model_input_size.height;

  // YuNet has 3 scales: Stride 8, 16, 32
  for (const int stride : {8, 16, 32}) {
    std::string s = std::to_string(stride);
    float *cls_ptr = m_cpu_buffers["cls_" + s].data();
    float *obj_ptr = m_cpu_buffers["obj_" + s].data();
    float *box_ptr = m_cpu_buffers["bbox_" + s].data();
    float *kps_ptr = m_cpu_buffers["kps_" + s].data();

    int feat_h = m_model_input_size.height / stride;
    int feat_w = m_model_input_size.width / stride;

    for (int i = 0; i < feat_h; ++i) {
      for (int j = 0; j < feat_w; ++j) {
        int idx = i * feat_w + j;
        const auto sigmoid = [](float x) {
          return 1.0f / (1.0f + std::exp(-x));
        };
        // const auto conf = sigmoid(cls_ptr[idx]) * sigmoid(obj_ptr[idx]);
        auto conf = cls_ptr[idx] * obj_ptr[idx];
        conf = sigmoid(cls_ptr[idx]) * sigmoid(obj_ptr[idx]);
        SPDLOG_INFO("Raw: {:.2f} | Sigmoid: {:.4f} | Final Conf: {:.4f}",
                    cls_ptr[idx], sigmoid(cls_ptr[idx]), conf);
        /*SPDLOG_INFO("conf w/ sigmoid: {}, conf w/o sigmoid: {}",
                    sigmoid(cls_ptr[idx]) * sigmoid(obj_ptr[idx]),
                    cls_ptr[idx] * obj_ptr[idx]);*/
        if (conf > m_score_threshold) {
          SPDLOG_INFO("(conf ({}) > m_score_threshold)", conf);
          FaceDetection face;
          face.confidence = conf;

          // 1. Box Decoding: [cx, cy, w, h] are offsets from anchor
          float cx =
              (static_cast<float>(j) + box_ptr[idx * 4 + 0]) * stride * sx;
          float cy =
              (static_cast<float>(i) + box_ptr[idx * 4 + 1]) * stride * sy;
          float w = std::exp(box_ptr[idx * 4 + 2]) * stride * sx;
          float h = std::exp(box_ptr[idx * 4 + 3]) * stride * sy;
          face.bbox = cv::Rect2f(cx - w / 2.0f, cy - h / 2.0f, w, h);

          // 2. Landmark Decoding: 5 points (x, y)
          for (int k = 0; k < 5; ++k) {
            // 2.1. Calculate the 'anchor' point (top-left of the grid cell)
            float anchor_x = static_cast<float>(j) * stride;
            float anchor_y = static_cast<float>(i) * stride;

            // 2.2. Add the raw offset (which is already in pixel-space)
            float raw_x = anchor_x + kps_ptr[idx * 10 + k * 2 + 0];
            float raw_y = anchor_y + kps_ptr[idx * 10 + k * 2 + 1];

            // 2.3. Finally, scale to your actual frame resolution
            face.landmarks[k].x = raw_x * sx;
            face.landmarks[k].y = raw_y * sy;
          }

          candidates.push_back(face);
          bboxes.push_back(face.bbox);
          scores.push_back(face.confidence);
        }
      }
    }
  }

  // 4. NMS is crucial because different scales might detect the same face
  std::vector<int> indices;
  cv::dnn::NMSBoxes(bboxes, scores, m_score_threshold, m_nms_threshold,
                    indices);

  ctx.yunet.clear();
  for (int idx : indices) {
    ctx.yunet.push_back(candidates[idx]);
  }
}

} // namespace MatrixPipeline::ProcessingUnit