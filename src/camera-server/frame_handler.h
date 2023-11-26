#ifndef CM_FRAME_HANDLER_H
#define CM_FRAME_HANDLER_H

#include "global_vars.h"

#include <opencv2/core/core.hpp>
#include <opencv2/cudacodec.hpp>

namespace FrameHandler {

void overlayDatetime(cv::Mat &frame, const double fontSacle,
                     const std::string &timestampOnDeviceOffline);

void overlayDeviceName(cv::Mat &frame, double fontSacle,
                       const std::string &deviceName);

void overlayContours(cv::Mat &dispFrame, cv::Mat &diffFrame);

void overlayStats(cv::Mat &frame, const float changeRate, const int cd,
                  const long long int videoFrameCount, const double fontSacle,
                  const enum MotionDetectionMode mode, const float currentFps,
                  const uint32_t maxFramesPerVideo);

float getFrameChanges(cv::cuda::GpuMat &prevFrame, cv::cuda::GpuMat &currFrame,
                      cv::cuda::GpuMat &diffFrame,
                      double pixelDiffAbsThreshold);

void generateBlankFrameAt1Fps(cv::cuda::GpuMat &currFrame,
                              const cv::Size &actualFrameSize);

} // namespace FrameHandler

#endif /* CM_FRAME_HANDLER_H */
