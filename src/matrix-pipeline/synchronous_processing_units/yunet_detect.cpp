#include "yunet_detect.h"

#include <opencv2/objdetect.hpp>

namespace MatrixPipeline::ProcessingUnit {

bool YuNetDetect::init(const njson &config) {
  try {
    const auto model_path = config.value("modelPath", "");
    m_score_threshold = config.value("scoreThreshold", m_score_threshold);
    // m_inference_interval =
    // std::chrono::milliseconds(config.value("inferenceIntervalMs",
    // m_inference_interval.count()));
    m_nms_threshold = config.value("nmsThreshold", m_nms_threshold);
    m_top_k = config.value("topK", m_top_k);

    if (model_path.empty()) {
      SPDLOG_ERROR("'modelPath' is missing in config");
      return false;
    }

    // Initialize OpenCV YuNet with an initial dummy size (1,1)
    m_detector = cv::FaceDetectorYN::create(
        model_path, "", cv::Size(1, 1), m_score_threshold, m_nms_threshold,
        m_top_k, cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA);

    SPDLOG_INFO("model_path: {}, score_threshold: {}, top_k: {}", model_path,
                m_score_threshold, m_top_k);
    return true;
  } catch (const std::exception &e) {
    SPDLOG_ERROR("e.what(): {}", e.what());
    return false;
  }
}

SynchronousProcessingResult YuNetDetect::process(cv::cuda::GpuMat &frame,
                                                 PipelineContext &ctx) {

  // m_last_inference_at = std::chrono::steady_clock::now();
  ctx.yunet_sface.results.clear();
  if (ctx.yunet_sface.results.size() > 0)
    throw std::runtime_error("");
  ctx.yunet_sface.yunet_input_frame_size = frame.size();
  if (m_detector->getInputSize() != frame.size()) {
    m_detector->setInputSize(frame.size());
  }

  frame.download(m_pinned_buffer);
  // point the cv::Mat directly to the pinned memory
  m_frame_cpu = m_pinned_buffer.createMatHeader();

  cv::Mat faces;
  m_detector->detect(m_frame_cpu, faces);

  if (!faces.empty()) {
    for (int i = 0; i < faces.rows; ++i) {
      YuNetDetection detection;
      // The raw output, we need this because cv::FaceRecognizerSF::alignCrop()
      // expects it as an input
      detection.yunet_output = faces.row(i);
      // [0-3]: Bounding Box (x, y, width, height)
      detection.bounding_box =
          cv::Rect2f(faces.at<float>(i, 0), faces.at<float>(i, 1),
                     faces.at<float>(i, 2), faces.at<float>(i, 3));

      // [4-13]: Landmarks (5 points)
      for (int j = 0; j < 5; ++j) {
        // FIXED: Using index 'j' for landmarks, not 'i'
        detection.landmarks[j] = {faces.at<float>(i, 4 + j * 2),
                                  faces.at<float>(i, 5 + j * 2)};
      }

      // [14]: Confidence score
      detection.confidence = faces.at<float>(i, 14);

      YuNetSFaceResult res;
      res.detection = std::move(detection);
      res.recognition = std::nullopt;
      ctx.yunet_sface.results.push_back(std::move(res));
    }
  }

  return success_and_continue;
}

} // namespace MatrixPipeline::ProcessingUnit