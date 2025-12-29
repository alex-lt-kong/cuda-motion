#include "yolo_detect.h"
#include "../utils/trt_logger.h"

#include <fmt/ranges.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
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
    m_model_path = config["modelPath"].get<std::string>();

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
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(Utils::g_logger));
    uint32_t flags =
        1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(flags));

    // 2. Parse ONNX
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, Utils::g_logger));
    if (!parser->parseFromFile(
            m_model_path.c_str(),
            static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
      SPDLOG_ERROR("Failed to parse ONNX file: {}", m_model_path);
      return false;
    }

    // 3. Build Engine
    auto trt_config = std::unique_ptr<nvinfer1::IBuilderConfig>(
        builder->createBuilderConfig());
    trt_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                   1ULL << 30); // 1GB

    SPDLOG_INFO("Building TensorRT Plan for model: {}...", m_model_path);
    SPDLOG_INFO("Note: Optimization trials may take several minutes depending "
                "on your GPU.");
    const auto start_time = std::chrono::steady_clock::now();
    const auto plan = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *trt_config));
    if (!plan) {
      SPDLOG_ERROR("builder->buildSerializedNetwork failed! Check your "
                   "TRTLogger for details.");
      return false;
    }
    const auto end_time = std::chrono::steady_clock::now();
    SPDLOG_INFO(
        "TensorRT Plan built successfully in {} seconds.",
        std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time)
            .count());
    const auto runtime = std::unique_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(Utils::g_logger));
    m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()));
    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(
        m_engine->createExecutionContext());

    // 4. Prepare Device Buffers
    // We assume index 0 is input, index 1 is output (standard for YOLO ONNX)
    nvinfer1::Dims output_dims =
        m_engine->getTensorShape(m_engine->getIOTensorName(1));
    // For YOLOv11, dims.d[1] is usually 84, dims.d[2] is 8400
    m_output_dimensions = output_dims.d[1];
    m_output_rows = output_dims.d[2];
    m_output_count = 1;
    for (int i = 0; i < output_dims.nbDims; ++i)
      m_output_count *= output_dims.d[i];
    cudaMalloc(&m_output_buffer_gpu, m_output_count * sizeof(float));
    m_output_cpu.resize(m_output_count);

    cudaStreamCreate(&m_cuda_stream);

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
  using namespace std::chrono;

  // 1. Interval Check
  const auto steady_now = std::chrono::steady_clock::now();
  if (steady_now - m_last_inference_time < m_inference_interval) {
    ctx.yolo = m_prev_yolo_ctx;
    return success_and_continue;
  }
  m_last_inference_time = steady_now;

  if (frame.empty() || !m_context) {
    return failure_and_continue;
  }

  // Clear previous results
  ctx.yolo.inference_outputs.clear(); // Not used by TRT but good to clean
  ctx.yolo.inference_input_size = m_model_input_size;

  try {
    // 2. Wrap the CUDA Stream
    cv::cuda::Stream cv_stream =
        cv::cuda::StreamAccessor::wrapStream(m_cuda_stream);

    // 3. Pre-processing (Thread-Safe with Local Buffers)
    // CRITICAL FIX: We use local variables instead of class members (like
    // m_gpu_resized) to prevent race conditions if multiple threads call
    // process() simultaneously.
    cv::cuda::GpuMat local_resized;
    cv::cuda::GpuMat local_input;

    cv::cuda::resize(frame, local_resized, m_model_input_size, 0, 0,
                     cv::INTER_LINEAR, cv_stream);
    // Convert to Float32 and Normalize (0-1 range) for YOLOv11
    // This creates the exact memory layout TensorRT expects.
    local_resized.convertTo(local_input, CV_32FC3, 1.0 / 255.0, cv_stream);

    // 4. Set TensorRT Bindings
    // We pass the pointer to our LOCAL buffer. This is safe only because we
    // synchronize before this function returns (and destroys local_input).
    m_context->setTensorAddress(m_engine->getIOTensorName(0), local_input.data);
    m_context->setTensorAddress(m_engine->getIOTensorName(1),
                                m_output_buffer_gpu);

    // 5. Enqueue Inference
    if (!m_context->enqueueV3(m_cuda_stream)) {
      SPDLOG_ERROR("TensorRT enqueueV3 failed");
      return failure_and_continue;
    }

    // 6. Copy Output (GPU -> CPU)
    // Moves data from m_output_buffer_gpu to m_output_cpu
    cudaMemcpyAsync(m_output_cpu.data(), m_output_buffer_gpu,
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
    // Disable inference temporarily on error to avoid log flooding
    m_inference_interval = std::chrono::milliseconds::max();
    return failure_and_continue;
  }
}

YoloDetect::~YoloDetect() {
  // 1. Raw GPU memory allocated via cudaMalloc
  if (m_output_buffer_gpu) {
    cudaFree(m_output_buffer_gpu);
  }

  // 2. CUDA Streams
  if (m_cuda_stream) {
    cudaStreamDestroy(m_cuda_stream);
  }

  // Note: m_engine and m_context are unique_ptrs,
  // so they clean themselves up automatically here!
}

} // namespace MatrixPipeline::ProcessingUnit