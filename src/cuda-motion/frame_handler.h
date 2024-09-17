#ifndef CM_FRAME_HANDLER_H
#define CM_FRAME_HANDLER_H

#include "global_vars.h"
#include "percentile_tracker.h"

#include <opencv2/core/core.hpp>
#include <opencv2/cudacodec.hpp>

#include <atomic>

namespace CudaMotion::Utils {

class FrameHandler {
  double fontScale;
  std::string deviceName;
  PercentileTracker<int64> pt;
  std::atomic<uint32_t> count;
  size_t overlayDatetimeBufLen;
  cv::Size overlayDatetimeTextSize;
  cv::Size overlayDeviceNameTextSize;

public:
  FrameHandler();
  FrameHandler(const double fontScale, const std::string &deviceName);
  void overlayDatetime(cv::Mat &frame,
                       const std::string &timestampOnDeviceOffline);

  void overlayDeviceName(cv::Mat &frame);

  void overlayContours(cv::Mat &dispFrame, cv::Mat &diffFrame);

  void overlayStats(cv::Mat &frame, const float changeRate, const int cd,
                    const long long int videoFrameCount, const float currentFps,
                    const uint32_t maxFramesPerVideo);

  float getFrameChanges(cv::cuda::GpuMat &prevFrame,
                        cv::cuda::GpuMat &currFrame,
                        cv::cuda::GpuMat &diffFrame,
                        double pixelDiffAbsThreshold);

  void rotate(cv::cuda::GpuMat &frame, const int frameRotationAngle);

  /**
   * @brief This acts like a guaranteed fallback video source:
   * vr->nextFrame(dCurrFrame) failed? fine, we will fill the frame with
   * nextDummyFrame() instead;
   */
  bool nextDummyFrame(cv::cuda::GpuMat &frame, const cv::Size &frameSize);
};
} // namespace CudaMotion::Utils

#endif /* CM_FRAME_HANDLER_H */
