#include "yunet_detect.h"

#include <opencv2/objdetect.hpp>

namespace MatrixPipeline::ProcessingUnit {

bool YuNetDetect::init(const njson &config) {
  try {
    // --- Translation Layer: camelCase JSON -> snake_case C++ ---
    const auto model_path = config.value("modelPath", "");
    m_score_threshold = config.value("scoreThreshold", m_score_threshold);
    m_inference_interval = std::chrono::milliseconds(
        config.value("inferenceIntervalMs", m_inference_interval.count()));
    m_nms_threshold = config.value("nmsThreshold", m_nms_threshold);
    m_top_k = config.value("topK", m_top_k);

    if (model_path.empty()) {
      SPDLOG_ERROR("'modelPath' is missing in config", m_unit_path);
      return false;
    }

    // Initialize OpenCV YuNet with an initial dummy size (1,1)
    m_detector = cv::FaceDetectorYN::create(
        model_path, "", cv::Size(1, 1), m_score_threshold, m_nms_threshold,
        m_top_k, cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA);

    SPDLOG_INFO("model_path: {}, inference_interval(ms): {}", model_path,
                m_inference_interval.count());
    return true;
  } catch (const std::exception &e) {
    SPDLOG_ERROR("e.what(): {}", e.what());
    m_disabled = true;
    return false;
  }
}

SynchronousProcessingResult YuNetDetect::process(cv::cuda::GpuMat &frame,
                                                 PipelineContext &ctx) {
  if (std::chrono::steady_clock::now() - m_last_inference_at <
      m_inference_interval) {
    ctx.yunet = m_prev_yunet_ctx;
    return success_and_continue;
  }
  m_last_inference_at = std::chrono::steady_clock::now();

  // 1. Dynamic Input Size Adjustment
  // Optimization: Only update if the resolution actually changed to avoid
  // buffer reallocation
  if (m_detector->getInputSize() != frame.size()) {
    m_detector->setInputSize(frame.size());
  }
  // 2. Optimized Download using Pinned Memory
  // This bypasses the extra internal copy the driver usually makes
  frame.download(m_pinned_buffer);

  // 3. Create a zero-copy Mat header
  // This creates a cv::Mat that points directly to the pinned memory
  cv::Mat h_frame = m_pinned_buffer.createMatHeader();
  // 4. Run Detection
  // OpenCV will still upload this, but the upload from pinned memory
  // is significantly faster and lower-latency in a VM.
  cv::Mat faces;
  m_detector->detect(h_frame, faces);

  // 3. Parse raw results into structured C++20/23 types
  YuNetContext detections;
  if (!faces.empty()) {
    detections.reserve(faces.rows); // Prevent multiple reallocations

    for (int i = 0; i < faces.rows; ++i) {
      FaceDetection face_data;
      face_data.face = faces.row(i);
      // [0-3]: Bounding Box (x, y, width, height)
      face_data.bbox = cv::Rect2f(faces.at<float>(i, 0), faces.at<float>(i, 1),
                                  faces.at<float>(i, 2), faces.at<float>(i, 3));

      // [4-13]: Landmarks (5 points)
      for (int j = 0; j < 5; ++j) {
        // FIXED: Using index 'j' for landmarks, not 'i'
        face_data.landmarks[j] = {faces.at<float>(i, 4 + j * 2),
                                  faces.at<float>(i, 5 + j * 2)};
      }

      // [14]: Confidence score
      face_data.confidence = faces.at<float>(i, 14);

      detections.push_back(std::move(face_data));
    }
  }

  // 4. Save to context
  ctx.yunet = std::move(detections);
  m_prev_yunet_ctx = ctx.yunet;
  return success_and_continue;
}

} // namespace MatrixPipeline::ProcessingUnit