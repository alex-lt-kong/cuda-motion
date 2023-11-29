#include "frame_handler.h"
#include "utils.h"

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <iomanip>

using namespace cv;
using namespace std;

namespace FrameHandler {

FrameHandler::FrameHandler(const double fontScale, const string &deviceName)
    : pt(10000) {
  count = 0;
  overlayDatetimeBufLen = UINT_MAX;
  this->fontScale = fontScale;
  this->deviceName = deviceName;
}

void FrameHandler::overlayDatetime(Mat &frame,
                                   const string &timestampOnDeviceOffline) {

  // This function is on the critical path and profiling shows it is slow
  // let's revert to good old C to make it faster!

  // strlen(19700101-000000) == 15
  time_t now;
  time(&now);
  char buf[15 * 4] = {0};
  strftime(buf, sizeof buf, "%Y%m%d-%H%M%S", localtime(&now));
  if (timestampOnDeviceOffline.size() > 0) [[unlikely]] {
    strcat(buf, " (Offline since ");
    strcat(buf, timestampOnDeviceOffline.c_str());
    strcat(buf, ")");
  }
  // int64 start = chrono::duration_cast<chrono::microseconds>(
  //                   chrono::system_clock::now().time_since_epoch())
  //                   .count();
  size_t buf_len = strlen(buf);
  if (overlayDatetimeBufLen != buf_len) {
    overlayDatetimeTextSize = getTextSize(buf, FONT_HERSHEY_DUPLEX, fontScale,
                                          8 * fontScale, nullptr);
    overlayDatetimeBufLen = buf_len;
  }

  putText(frame, buf, Point(5, overlayDatetimeTextSize.height * 1.05),
          FONT_HERSHEY_DUPLEX, fontScale, Scalar(0, 0, 0), 8 * fontScale,
          LINE_8, false);
  putText(frame, buf, Point(5, overlayDatetimeTextSize.height * 1.05),
          FONT_HERSHEY_DUPLEX, fontScale, Scalar(255, 255, 255), 2 * fontScale,
          LINE_8, false);
  // int64 end = chrono::duration_cast<chrono::microseconds>(
  //                 chrono::system_clock::now().time_since_epoch())
  //                 .count();
  //++count;
  // pt.addNumber(end - start);
  // if (count % 1000 == 0) {
  // pt.refreshStats();
  // double percentiles[] = {50, 66, 90, 95, 100};
  // spdlog::info("Percentiles (sampleCount: {})", pt.sampleCount());
  // for (size_t i = 0; i < sizeof(percentiles) / sizeof(percentiles[0]); ++i) {
  //   spdlog::info("{:3}-th: {:6}", percentiles[i],
  //                pt.getPercentile(percentiles[i]));
  // }
  // Percentiles (sampleCount: 10000)
  //  50-th:   1844
  //  66-th:   2141
  //  90-th:   4915
  //  95-th:   9989
  // 100-th: 133689
}

void FrameHandler::overlayDeviceName(Mat &frame) {

  Size textSize = getTextSize(deviceName, FONT_HERSHEY_DUPLEX, fontScale,
                              8 * fontScale, nullptr);
  putText(frame, deviceName,
          Point(frame.cols - textSize.width * 1.05, frame.rows - 5),
          FONT_HERSHEY_DUPLEX, fontScale, Scalar(0, 0, 0), 8 * fontScale,
          LINE_8, false);
  putText(frame, deviceName,
          Point(frame.cols - textSize.width * 1.05, frame.rows - 5),
          FONT_HERSHEY_DUPLEX, fontScale, Scalar(255, 255, 255), 2 * fontScale,
          LINE_8, false);
}

void FrameHandler::overlayContours(Mat &dispFrame, Mat &diffFrame) {
  if (diffFrame.empty())
    return;
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;

  findContours(diffFrame, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

  for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
    drawContours(dispFrame, contours, idx, Scalar(255, 255, 255), 1, LINE_8,
                 hierarchy);
  }
}

void FrameHandler::overlayStats(Mat &frame, const float changeRate,
                                const int cd,
                                const long long int videoFrameCount,
                                const enum MotionDetectionMode mode,
                                const float currentFps,
                                const uint32_t maxFramesPerVideo) {
  // Profiling shows this function is a performance-critical one, reverting to
  // C gives us much better performance
  char buff[256] = {0};
  snprintf(buff, sizeof(buff) / sizeof(buff[0]) - 1,
           "%.2f%%, %.1ffps (%d, %lld)", changeRate, currentFps, cd,
           maxFramesPerVideo - videoFrameCount);
  /*ostringstream oss;
  if (mode == MODE_DETECT_MOTION) {
    oss << fixed << setprecision(2) << changeRate << "%, ";
  }
  oss << fixed << setprecision(1) << currentFps << "fps ";
  if (mode != MODE_DISABLED) {
    oss << "(" << cd << ", " << maxFramesPerVideo - videoFrameCount << ")";
  }*/
  putText(frame, buff, Point(5, frame.rows - 5), FONT_HERSHEY_DUPLEX, fontScale,
          Scalar(0, 0, 0), 8 * fontScale, LINE_8, false);
  putText(frame, buff, Point(5, frame.rows - 5), FONT_HERSHEY_DUPLEX, fontScale,
          Scalar(255, 255, 255), 2 * fontScale, LINE_8, false);
}

float FrameHandler::getFrameChanges(cv::cuda::GpuMat &prevFrame,
                                    cv::cuda::GpuMat &currFrame,
                                    cv::cuda::GpuMat &diffFrame,
                                    double pixelDiffAbsThreshold) {
  if (prevFrame.empty() || currFrame.empty()) {
    return -1;
  }
  if (prevFrame.cols != currFrame.cols || prevFrame.rows != currFrame.rows) {
    return -1;
  }
  if (prevFrame.cols == 0 || prevFrame.rows == 0) {
    return -1;
  }

  cuda::absdiff(prevFrame, currFrame, diffFrame);
  cuda::cvtColor(diffFrame, diffFrame, COLOR_BGR2GRAY);
  cuda::threshold(diffFrame, diffFrame, pixelDiffAbsThreshold, 255,
                  THRESH_BINARY);
  int nonZeroPixels = cuda::countNonZero(diffFrame);
  return 100.0 * nonZeroPixels / (diffFrame.rows * diffFrame.cols);
}

void FrameHandler::generateBlankFrameAt1Fps(cv::cuda::GpuMat &currFrame,
                                            const Size &actualFrameSize) {
  this_thread::sleep_for(999ms); // Throttle the generation at 1 fps.

  /* Even if we generate nothing but a blank screen, we cant just use some
  hardcoded values and skip framepreferredInputWidth/actualFrameSize.width and
  framepreferredInputHeight/actualFrameSize.height.
  The problem will occur when piping frames to ffmpeg: In ffmpeg, we
  pre-define the frame size, which is mostly framepreferredInputWidth x
  framepreferredInputHeight. If the video device is down and we supply a
  smaller frame, ffmpeg will wait until there are enough pixels filling
  the original resolution to write one frame, causing screen tearing
  */
  if (actualFrameSize.width > 0 && actualFrameSize.height > 0) {
    currFrame = cv::cuda::GpuMat(actualFrameSize.height, actualFrameSize.width,
                                 CV_8UC3, Scalar(128, 128, 128));
  } else {
    currFrame = cv::cuda::GpuMat(540, 960, CV_8UC3, Scalar(128, 128, 128));
    // 960x540, 1280x760, 1920x1080 all have 16:9 aspect ratio.
  }
}
} // namespace FrameHandler
