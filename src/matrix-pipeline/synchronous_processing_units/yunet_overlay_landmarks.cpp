#include "yunet_overlay_landmarks.h"
#include <opencv2/imgproc.hpp>

namespace MatrixPipeline::ProcessingUnit {

bool YuNetOverlayLandmarks::init(const njson &config) {
  try {
    if (const auto landmark_color_bgr_key = "landmarkColorBgr";
        config.contains(landmark_color_bgr_key)) {
      // Note the constructor of cv::Scalar(), the alternative way (of using
      // njson's default value support) is equally awkward
      m_landmark_color_bgr = cv::Scalar(config[landmark_color_bgr_key][0],
                                        config[landmark_color_bgr_key][1],
                                        config[landmark_color_bgr_key][2]);
    }
    m_radius = config.value("radius", 2);
    m_thickness = config.value("thickness", -1);
    SPDLOG_INFO("radius: {}, thickness: {}, landmark_color_bgr: {{{}, {}, {}}}",
                m_radius, m_thickness, m_landmark_color_bgr[0],
                m_landmark_color_bgr[1], m_landmark_color_bgr[2]);
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