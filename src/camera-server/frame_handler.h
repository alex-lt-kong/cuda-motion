#ifndef CS_FRAME_HANDLER_H
#define CS_FRAME_HANDLER_H

#include "global_vars.h"

#include <opencv2/core/core.hpp>

namespace FrameHandler {

void overlayDatetime(cv::Mat &frame, double fontSacle,
                     std::string timestampOnDeviceOffline);

void overlayDeviceName(cv::Mat &frame, double fontSacle,
                       std::string deviceName);

void overlayContours(cv::Mat &dispFrame, cv::Mat &diffFrame);

void overlayStats(cv::Mat &frame, float changeRate, int cd,
                  long long int videoFrameCount, double fontSacle,
                  enum MotionDetectionMode mode, float currentFps,
                  uint32_t maxFramesPerVideo);

float getFrameChanges(cv::Mat &prevFrame, cv::Mat &currFrame,
                      cv::Mat *diffFrame, double pixelDiffAbsThreshold);

} // namespace FrameHandler

#endif /* CS_FRAME_HANDLER_H */
