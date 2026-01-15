#include "yolo_detect.h"
#include "../utils/cuda_helper.h"

#include <fmt/ranges.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/dnn.hpp>
#include <spdlog/spdlog.h>

#include <string>
#include <vector>

namespace MatrixPipeline::ProcessingUnit {

bool YoloDetect::init(const njson &config) {
  try {
    if (!config.contains("modelPath")) {
      SPDLOG_ERROR("modelPath not defined");
      return false;
    }
    const auto model_path = config["modelPath"].get<std::string>();

    // Allow overriding input size via config
    if (config.contains("inputWidth"))
      m_model_input_size.width = config["inputWidth"].get<int>();
    if (config.contains("inputHeight"))
      m_model_input_size.height = config["inputHeight"].get<int>();
    m_inference_interval = std::chrono::milliseconds(
        config.value("inferenceIntervalMs", m_inference_interval.count()));
    m_confidence_threshold =
        config.value("confidenceThreshold", m_confidence_threshold);

    // 1. Initialize TensorRT Builder and Network
    const auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(Utils::g_logger));
    constexpr auto flags =
        1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    const auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(flags));

    // 2. Parse ONNX
    const auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, Utils::g_logger));
    if (!parser->parseFromFile(
            model_path.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
      SPDLOG_ERROR("Failed to parse ONNX file: {}", model_path);
      return false;
    }

    // 3. Build Engine
    auto trt_config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    trt_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                   1ULL << 30); // 1GB

    SPDLOG_INFO("Building TensorRT Plan for model: {}...", model_path);
    SPDLOG_INFO("Note: Optimization trials may take several minutes depending "
                "on your GPU.");
    const auto start_time = std::chrono::steady_clock::now();
    const auto plan = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *trt_config));
    if (!plan) {
      SPDLOG_ERROR("builder->buildSerializedNetwork failed");
      return false;
    }
    const auto end_time = std::chrono::steady_clock::now();
    SPDLOG_INFO(
        "TensorRT Plan built successfully in {} seconds.",
        std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time)
            .count());
    const auto runtime = std::unique_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(Utils::g_logger));
    // Much of the above boilerplate code is using local variables only, the
    // real things we need for inference are just m_engine and m_context
    m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(
        m_engine->createExecutionContext());

    // 4. Prepare Device Buffers
    // We assume index 0 is input, index 1 is output (standard for YOLO ONNX)
    const auto output_dims =
        m_engine->getTensorShape(m_engine->getIOTensorName(1));
    // For YOLOv11, dims.d[1] is usually 84, dims.d[2] is 8400
    m_output_dimensions = output_dims.d[1];
    m_output_rows = output_dims.d[2];
    {
      // m_output_count is total number of scalar elements (floats) in the
      // tensor if you were to lay them all out in a single straight line. Say
      // your tensor's dimensions are [1, 84, 8400], m_output_count will be 1
      // * 84 * 8400 = 705600
      m_output_count = 1;
      for (int i = 0; i < output_dims.nbDims; ++i)
        m_output_count *= output_dims.d[i];
      m_output_buffer_gpu =
          Utils::make_device_unique<float>(m_output_count * sizeof(float));
      m_output_cpu.resize(m_output_count);
    }

    m_input_buffer_gpu = Utils::make_device_unique<float>(
        3 * m_model_input_size.width * m_model_input_size.height *
        sizeof(float));

    {
      if (cudaStreamCreate(&m_cuda_stream) != cudaSuccess) {
        SPDLOG_ERROR("cudaStreamCreate() failed");
        return false;
      }
      m_cv_stream = cv::cuda::StreamAccessor::wrapStream(m_cuda_stream);
    }

    {
      // 1. Verify Count (Must be SISO)
      if (m_engine->getNbIOTensors() != 2) {
        throw std::runtime_error(
            "Error: Model must have exactly 1 Input and 1 Output.");
      }

      // 2. Verify Index 0 is INPUT
      const char *input_name = m_engine->getIOTensorName(0);
      if (m_engine->getTensorIOMode(input_name) !=
          nvinfer1::TensorIOMode::kINPUT) {
        throw std::runtime_error(
            "Error: Tensor at Index 0 must be INPUT (images).");
      }

      // 3. Verify Index 1 is OUTPUT
      const char *output_name = m_engine->getIOTensorName(1);
      if (m_engine->getTensorIOMode(output_name) !=
          nvinfer1::TensorIOMode::kOUTPUT) {
        throw std::runtime_error("Error: Tensor at Index 1 must be OUTPUT.");
      }

      // 4. If we survived, bind them strictly
      m_context->setTensorAddress(input_name, m_input_buffer_gpu.get());
      m_context->setTensorAddress(output_name, m_output_buffer_gpu.get());
    }

    SPDLOG_INFO("TensorRT Engine initialized from ONNX. Output size: {}",
                m_output_count);
    return true;
  } catch (const std::exception &e) {
    SPDLOG_ERROR("Init failed: {}", e.what());
    return false;
  }
}

void YoloDetect::post_process_yolo(const cv::cuda::GpuMat &frame,
                                   PipelineContext &ctx) const {
  // 1. Calculate Scale Factor (Frame vs Input Blob)
  const double x_factor =
      static_cast<double>(frame.cols) / m_model_input_size.width;
  const double y_factor =
      static_cast<double>(frame.rows) / m_model_input_size.height;

  // 2. Wrap the CPU Output Buffer
  // We use the dimensions calculated in init() (e.g., 84 x 8400)
  // m_output_cpu contains the data copied from GPU in process()
  cv::Mat result_wrapper(m_output_dimensions, m_output_rows, CV_32F,
                         const_cast<float *>(m_output_cpu.data()));
  cv::Mat output_t;
  cv::transpose(result_wrapper,
                output_t); // Transpose to [rows, dimensions] (e.g. [8400, 84])

  // 3. Reset Context Data
  ctx.yolo.class_ids.clear();
  ctx.yolo.confidences.clear();
  ctx.yolo.boxes.clear();
  ctx.yolo.is_detection_valid.clear(); // Important to clear this too

  // 4. Iterate over rows (anchors)
  // m_output_rows is typically 8400 for YOLOv11
  for (int i = 0; i < m_output_rows; ++i) {
    auto *row_ptr = output_t.ptr<float>(i);

    // Scores start at index 4 (after cx, cy, w, h)
    // length is dimensions - 4 (e.g., 84 - 4 = 80 classes)
    cv::Mat scores(1, m_output_dimensions - 4, CV_32F, row_ptr + 4);

    cv::Point class_id_point;
    double max_class_score;
    cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);

    if (max_class_score > m_confidence_threshold) {
      float cx = row_ptr[0];
      float cy = row_ptr[1];
      float w = row_ptr[2];
      float h = row_ptr[3];

      auto left = static_cast<int>((cx - 0.5 * w) * x_factor);
      auto top = static_cast<int>((cy - 0.5 * h) * y_factor);
      auto width = static_cast<int>(w * x_factor);
      auto height = static_cast<int>(h * y_factor);

      ctx.yolo.boxes.emplace_back(left, top, width, height);
      ctx.yolo.confidences.push_back(static_cast<float>(max_class_score));
      ctx.yolo.class_ids.push_back(class_id_point.x);
      ctx.yolo.is_detection_valid.push_back(false);
    }
  }

  // 5. NMS
  ctx.yolo.indices.clear();
  cv::dnn::NMSBoxes(ctx.yolo.boxes, ctx.yolo.confidences,
                    m_confidence_threshold, m_nms_thres, ctx.yolo.indices);
}

SynchronousProcessingResult YoloDetect::process(cv::cuda::GpuMat &frame,
                                                PipelineContext &ctx) {
  struct LetterboxProps {
    float scale;
    int x_offset;
    int y_offset;
  };

  // 2. The Lambda Definition
  auto letterbox_resize =
      [](const cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst,
         cv::cuda::GpuMat
             &intermediate_buffer, // Optimized: Pass pre-allocated buffer
         const cv::Size &target_size,
         cv::cuda::Stream &stream) -> LetterboxProps {
    // A. Calculate Scaling Ratio (Model / Input)
    const float scale_x = static_cast<float>(target_size.width) / src.cols;
    const float scale_y = static_cast<float>(target_size.height) / src.rows;
    const float scale = std::min(scale_x, scale_y);

    // B. Calculate New Dimensions (Unpadded)
    const int new_w = static_cast<int>(std::round(src.cols * scale));
    const int new_h = static_cast<int>(std::round(src.rows * scale));

    // C. Resize into the intermediate buffer (Standard Linear Resize)
    // Note: We use the passed 'intermediate_buffer' to avoid malloc on every
    // frame
    cv::cuda::resize(src, intermediate_buffer, cv::Size(new_w, new_h), 0, 0,
                     cv::INTER_LINEAR, stream);

    // D. Prepare Destination (Black/Grey Padding)
    if (dst.size() != target_size || dst.type() != src.type()) {
      dst.create(target_size, src.type());
    }
    // 114 is the standard YOLO grey, 0 is black. Changing to 114 is safer for
    // accuracy.
    dst.setTo(cv::Scalar(114, 114, 114), stream);
    // E. Copy Resized Image to Center of Destination
    const int x_offset = (target_size.width - new_w) / 2;
    const int y_offset = (target_size.height - new_h) / 2;

    cv::cuda::GpuMat dst_roi = dst(cv::Rect(x_offset, y_offset, new_w, new_h));
    intermediate_buffer.copyTo(dst_roi, stream);

    // F. Return transformation properties for Post-Processing NMS
    return LetterboxProps{scale, x_offset, y_offset};
  };

  using namespace std::chrono;

  const auto steady_now = steady_clock::now();
  if (steady_now - m_last_inference_time < m_inference_interval) {
    ctx.yolo = m_prev_yolo_ctx;
    return success_and_continue;
  }
  m_last_inference_time = steady_now;

  if (frame.empty() || !m_context) {
    return failure_and_continue;
  }

  ctx.yolo.inference_input_size = m_model_input_size;

  try {
    // YOLO expects us to use "letterbox resize", not just resize()
    letterbox_resize(frame, m_resized_gpu, m_resized_gpu_buffer,
                     m_model_input_size, m_cv_stream);

    // 2. Color Convert: BGR -> RGB
    cv::cuda::cvtColor(m_resized_gpu, m_rgb, cv::COLOR_BGR2RGB, 0, m_cv_stream);
    // alpha is  1.0/255.0, while in yunet_detect.cpp it is 1.0
    // Convert to Float32 and Normalize (0-1 range) for YOLOv11
    // This creates the exact memory layout TensorRT expects.
    m_rgb.convertTo(m_normalized_gpu, CV_32FC3, 1.0 / 255.0, m_cv_stream);

    // 4. HWC -> NCHW Conversion， We split the interleaved Mat into 3 separate
    // planes (R, G, B)
    std::vector<cv::cuda::GpuMat> channels;
    cv::cuda::split(m_normalized_gpu, channels, m_cv_stream);

    for (int i = 0; i < 3; ++i) {
      // We use cudaMemcpy2DAsync because GpuMat rows might be padded (step !=
      // width)
      cudaMemcpy2DAsync(
          m_input_buffer_gpu.get() +
              (i *
               m_model_input_size.area()), // Dest: Offset for R, G, or B plane
          m_model_input_size.width *
              sizeof(float), // Dest Pitch (Linear, so equal to width)
          channels[i].data,  // Src: Ptr to GpuMat data
          channels[i].step,  // Src Pitch: GpuMat step (padding)
          m_model_input_size.width * sizeof(float), // Width in bytes to copy
          m_model_input_size.height,                // Height (rows)
          cudaMemcpyDeviceToDevice, m_cuda_stream);
    }

    // 5. Enqueue Inference
    if (!m_context->enqueueV3(m_cuda_stream)) {
      SPDLOG_ERROR("TensorRT enqueueV3 failed");
      return failure_and_continue;
    }

    // 6. Copy Output (GPU -> CPU)
    // Moves data from m_output_buffer_gpu to m_output_cpu
    cudaMemcpyAsync(m_output_cpu.data(), m_output_buffer_gpu.get(),
                    m_output_count * sizeof(float), cudaMemcpyDeviceToHost,
                    m_cuda_stream);

    // 7. Synchronize
    // This effectively "locks" the CPU until inference is done.
    // It prevents 'local_input' from being destroyed while GPU is still reading
    // it.
    cudaStreamSynchronize(m_cuda_stream);

    // 8. Parse Results
    post_process_yolo(frame, ctx);

    m_prev_yolo_ctx = ctx.yolo;
    return success_and_continue;

  } catch (const std::exception &e) {
    SPDLOG_ERROR("Inference Error: {}", e.what());
    disable();
    return failure_and_continue;
  }
}

YoloDetect::~YoloDetect() {

  if (m_cuda_stream) {
    cudaStreamDestroy(m_cuda_stream);
  }
}

} // namespace MatrixPipeline::ProcessingUnit