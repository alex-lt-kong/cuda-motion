#include "yolo_detect.h"

#include <fmt/ranges.h>
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
    SPDLOG_INFO("Loading ONNX model: {}", m_model_path);

    m_net = cv::dnn::readNetFromONNX(m_model_path);
    m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    if (m_net.empty()) {
      SPDLOG_ERROR("Failed to load ONNX model: {}", m_model_path);
      return false;
    }
    SPDLOG_INFO("model_path: {}, inference_interval(ms): {}, "
                "confidence_threshold: {}, model_input_size: {}x{}",
                m_model_path, m_inference_interval.count(),
                m_confidence_threshold, m_model_input_size.width,
                m_model_input_size.height);
    return true;
  } catch (const std::exception &e) {
    SPDLOG_ERROR("Init failed: {}", e.what());
    return false;
  }
}

void YoloDetect::post_process_yolo(const cv::cuda::GpuMat &frame,
                                   PipelineContext &ctx) const {
  const std::vector<cv::Mat> &outputs = ctx.yolo.inference_outputs;

  // Calculate Scale Factor (Frame vs Input Blob)
  const double x_factor =
      static_cast<double>(frame.cols) / ctx.yolo.inference_input_size.width;
  const double y_factor =
      static_cast<double>(frame.rows) / ctx.yolo.inference_input_size.height;

  // --- YOLO Output Parsing ---
  // (This logic is specific to YOLOv5/v8 flattened output)
  const cv::Mat &output = outputs[0];
  int dimensions = output.size[1];
  int rows = output.size[2];

  cv::Mat result_wrapper(dimensions, rows, CV_32F, output.data);
  cv::Mat output_t;
  cv::transpose(result_wrapper, output_t); // Transpose to [rows, dimensions]

  ctx.yolo.class_ids.clear();
  ctx.yolo.confidences.clear();
  ctx.yolo.boxes.clear();

  for (int i = 0; i < rows; ++i) {
    auto *row_ptr = output_t.ptr<float>(i);
    cv::Mat scores(1, dimensions - 4, CV_32F, row_ptr + 4);
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

  ctx.yolo.indices.clear();
  cv::dnn::NMSBoxes(ctx.yolo.boxes, ctx.yolo.confidences,
                    m_confidence_threshold, m_nms_thres, ctx.yolo.indices);
}

SynchronousProcessingResult YoloDetect::process(cv::cuda::GpuMat &frame,
                                                PipelineContext &ctx) {
  using namespace std::chrono;
  const auto steady_now = std::chrono::steady_clock::now();
  if (steady_now - m_last_inference_time < m_inference_interval) {
    ctx.yolo = m_prev_yolo_ctx;
    return success_and_continue;
  }

  m_last_inference_time = steady_now;
  if (frame.empty() || m_net.empty())
    return failure_and_continue;

  ctx.yolo.inference_outputs.clear();
  ctx.yolo.inference_input_size = m_model_input_size;

  try {
    // 1. Pre-process: Resize and Color Convert on GPU
    cv::cuda::GpuMat resized_rgb;
    cv::cuda::resize(frame, resized_rgb, m_model_input_size);
    cv::cuda::cvtColor(resized_rgb, resized_rgb, cv::COLOR_BGR2RGB);

    // 2. GPU-Side Normalization (Replaces CPU blobFromImage math)
    // Convert to 32-bit float and scale by 1/255.0 directly on the GPU
    cv::cuda::GpuMat normalized_gpu;
    resized_rgb.convertTo(normalized_gpu, CV_32F, 1.0 / 255.0);

    // 3. Optimized Download
    // Now we only download the final, processed pixels.
    cv::Mat h_blob_ready;
    normalized_gpu.download(h_blob_ready);

    // 4. Create Blob from already-normalized CPU data
    // Since data is already RGB and scaled, we pass 1.0 scale and no mean
    // subtraction.
    cv::Mat blob = cv::dnn::blobFromImage(h_blob_ready, 1.0, cv::Size(),
                                          cv::Scalar(), false, false);

    m_net.setInput(blob);

    // Populate the vector in the context directly
    m_net.forward(ctx.yolo.inference_outputs,
                  m_net.getUnconnectedOutLayersNames());

    // 3. Decode & Visualize (If inference produced data)
    if (!ctx.yolo.inference_outputs.empty()) {
      post_process_yolo(frame, ctx);
    }
    m_prev_yolo_ctx = ctx.yolo;
    return success_and_continue;

  } catch (const cv::Exception &e) {
    SPDLOG_ERROR("Inference Error: {}", e.what());
    m_inference_interval = std::chrono::milliseconds::max();
    SPDLOG_WARN("Inference is disabled to prevent flooding of log");
    return failure_and_continue;
  }
}

} // namespace MatrixPipeline::ProcessingUnit