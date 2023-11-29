#ifndef CM_FRAME_HANDLER_H
#define CM_FRAME_HANDLER_H

#include "global_vars.h"
#include "percentile_tracker.h"

#include <opencv2/core/core.hpp>
#include <opencv2/cudacodec.hpp>

#include <atomic>

namespace FrameHandler {

class FrameHandler {
  double fontScale;
  std::string deviceName;
  PercentileTracker<int64> pt;
  std::atomic<uint32_t> count;
  size_t overlayDatetimeBufLen;
  cv::Size overlayDatetimeTextSize;

public:
  FrameHandler();
  FrameHandler(const double fontScale, const std::string &deviceName);
  void overlayDatetime(cv::Mat &frame,
                       const std::string &timestampOnDeviceOffline);

  void overlayDeviceName(cv::Mat &frame);

  void overlayContours(cv::Mat &dispFrame, cv::Mat &diffFrame);

  void overlayStats(cv::Mat &frame, const float changeRate, const int cd,
                    const long long int videoFrameCount,
                    const enum MotionDetectionMode mode, const float currentFps,
                    const uint32_t maxFramesPerVideo);

  float getFrameChanges(cv::cuda::GpuMat &prevFrame,
                        cv::cuda::GpuMat &currFrame,
                        cv::cuda::GpuMat &diffFrame,
                        double pixelDiffAbsThreshold);

  void generateBlankFrameAt1Fps(cv::cuda::GpuMat &currFrame,
                                const cv::Size &actualFrameSize);
};
} // namespace FrameHandler

#endif /* CM_FRAME_HANDLER_H */
