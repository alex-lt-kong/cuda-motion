#include "yunet_overlay_landmarks.h"
#include <opencv2/imgproc.hpp>

namespace MatrixPipeline::ProcessingUnit {

bool YuNetOverlayLandmarks::init(const njson &config) {
  try {
    if (config.contains("m  ")) {
      // Note the constructor of cv::Scalar(), the alternative way is equally
      // awkward
      auto color = config["landmarkColorBgr"];
      m_landmark_color_bgr = cv::Scalar(color[0], color[1], color[2]);
    }
    m_radius = config.value("radius", 2);
    m_thickness = config.value("thickness", -1);
    SPDLOG_INFO("radius: {}, thickness: {}", m_radius, m_thickness);
    return true;
  } catch (const std::exception &e) {
    SPDLOG_ERROR("Failed to init YuNetOverlayLandmarks: {}", e.what());
    return false;
  }
}

SynchronousProcessingResult
YuNetOverlayLandmarks::process(cv::cuda::GpuMat &frame, PipelineContext &ctx) {
  // 1. Check if YuNet data exists in context
  if (ctx.yunet.empty()) {
    return SynchronousProcessingResult::failure_and_continue;
  }

  // 2. OpenCV drawing functions require CPU Mat.
  // We download the frame to CPU to draw the overlays.
  cv::Mat cpu_frame;
  frame.download(cpu_frame);

  // 3. Iterate through detected faces and draw landmarks
  for (const auto &face : ctx.yunet) {
    for (const auto &landmark : face.landmarks) {
      // Draw a circle for each of the 5 points (eyes, nose, mouth corners)
      cv::circle(cpu_frame, landmark, m_radius, m_landmark_color_bgr,
                 m_thickness);
    }
  }

  // 4. Upload the modified frame back to GPU
  frame.upload(cpu_frame);

  return SynchronousProcessingResult::success_and_continue;
}

} // namespace MatrixPipeline::ProcessingUnit