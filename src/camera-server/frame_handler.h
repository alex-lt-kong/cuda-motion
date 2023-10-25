#ifndef CS_FRAME_HANDLER_H
#define CS_FRAME_HANDLER_H

#include "global_vars.h"

#include <opencv2/core/core.hpp>

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

float getFrameChanges(cv::Mat &prevFrame, cv::Mat &currFrame,
                      cv::Mat *diffFrame, double pixelDiffAbsThreshold);

void generateBlankFrameAt1Fps(cv::Mat &currFrame,
                              const cv::Size &actualFrameSize);

} // namespace FrameHandler

#endif /* CS_FRAME_HANDLER_H */
