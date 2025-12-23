#include "yunet_detect.h"

#include <opencv2/objdetect.hpp>

namespace MatrixPipeline::ProcessingUnit {

bool YuNetDetect::init(const njson &config) {
  try {
    // --- Translation Layer: camelCase JSON -> snake_case C++ ---
    std::string model_path = config.value("modelPath", "");
    m_score_threshold = config.value("scoreThreshold", m_score_threshold);
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

    SPDLOG_INFO("initialized successfully from {}", m_unit_path, model_path);
    return true;
  } catch (const std::exception &e) {
    SPDLOG_ERROR("e.what(): {}", e.what());
    m_disabled = true;
    return false;
  }
}

SynchronousProcessingResult YuNetDetect::process(cv::cuda::GpuMat &gpu_frame,
                                                 PipelineContext &ctx) {
  if (m_disabled)
    return failure_and_continue;
  // 1. Dynamic Input Size Adjustment
  // Optimization: Only update if the resolution actually changed to avoid
  // buffer reallocation
  if (m_detector->getInputSize() != gpu_frame.size()) {
    m_detector->setInputSize(gpu_frame.size());
  }
  cv::Mat h_frame;
  gpu_frame.download(h_frame);
  // 2. Inference (GPU-to-GPU)
  // Because m_detector was initialized with DNN_BACKEND_CUDA,
  // passing a GpuMat here keeps the frame on the device.
  cv::Mat faces;
  m_detector->detect(h_frame, faces);

  // 3. Parse raw results into structured C++20/23 types
  YuNetContext detections;
  if (!faces.empty()) {
    detections.reserve(faces.rows); // Prevent multiple reallocations

    for (int i = 0; i < faces.rows; ++i) {
      FaceDetection face_data;

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

  return success_and_continue;
}

} // namespace MatrixPipeline::ProcessingUnit