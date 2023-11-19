#include "frame_handler.h"
#include "utils.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iomanip>

using namespace cv;
using namespace std;

namespace FrameHandler {

void overlayDatetime(Mat &frame, const double fontSacle,
                     const string &timestampOnDeviceOffline) {
  time_t now;
  time(&now);
  // char buf[sizeof "1970-01-01 00:00:00"];
  // strftime(buf, sizeof buf, "%F %T", localtime(&now));
  string ts = getCurrentTimestamp();
  if (timestampOnDeviceOffline.size() > 0) {
    ts += " (Offline since " + timestampOnDeviceOffline + ")";
  }
  cv::Size textSize =
      getTextSize(ts, FONT_HERSHEY_DUPLEX, fontSacle, 8 * fontSacle, nullptr);
  putText(frame, ts, Point(5, textSize.height * 1.05), FONT_HERSHEY_DUPLEX,
          fontSacle, Scalar(0, 0, 0), 8 * fontSacle, LINE_8, false);
  putText(frame, ts, Point(5, textSize.height * 1.05), FONT_HERSHEY_DUPLEX,
          fontSacle, Scalar(255, 255, 255), 2 * fontSacle, LINE_8, false);
}

void overlayDeviceName(Mat &frame, const double fontSacle,
                       const string &deviceName) {

  cv::Size textSize = getTextSize(deviceName, FONT_HERSHEY_DUPLEX, fontSacle,
                                  8 * fontSacle, nullptr);
  putText(frame, deviceName,
          Point(frame.cols - textSize.width * 1.05, frame.rows - 5),
          FONT_HERSHEY_DUPLEX, fontSacle, Scalar(0, 0, 0), 8 * fontSacle,
          LINE_8, false);
  putText(frame, deviceName,
          Point(frame.cols - textSize.width * 1.05, frame.rows - 5),
          FONT_HERSHEY_DUPLEX, fontSacle, Scalar(255, 255, 255), 2 * fontSacle,
          LINE_8, false);
}

void overlayContours(Mat &dispFrame, Mat &diffFrame) {
  if (diffFrame.empty())
    return;
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;

  findContours(diffFrame, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

  for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
    cv::drawContours(dispFrame, contours, idx, Scalar(255, 255, 255), 1, LINE_8,
                     hierarchy);
  }
}

void overlayStats(Mat &frame, const float changeRate, const int cd,
                  const long long int videoFrameCount, const double fontSacle,
                  const enum MotionDetectionMode mode, const float currentFps,
                  const uint32_t maxFramesPerVideo) {
  ostringstream oss;
  if (mode == MODE_DETECT_MOTION) {
    oss << fixed << setprecision(2) << changeRate << "%, ";
  }
  oss << fixed << setprecision(1) << currentFps << "fps ";
  if (mode != MODE_DISABLED) {
    oss << "(" << cd << ", " << maxFramesPerVideo - videoFrameCount << ")";
  }
  putText(frame, oss.str(), Point(5, frame.rows - 5), FONT_HERSHEY_DUPLEX,
          fontSacle, Scalar(0, 0, 0), 8 * fontSacle, LINE_8, false);
  putText(frame, oss.str(), Point(5, frame.rows - 5), FONT_HERSHEY_DUPLEX,
          fontSacle, Scalar(255, 255, 255), 2 * fontSacle, LINE_8, false);
}

float getFrameChanges(Mat &prevFrame, Mat &currFrame, Mat *diffFrame,
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

  absdiff(prevFrame, currFrame, *diffFrame);
  cvtColor(*diffFrame, *diffFrame, COLOR_BGR2GRAY);
  threshold(*diffFrame, *diffFrame, pixelDiffAbsThreshold, 255, THRESH_BINARY);
  int nonZeroPixels = countNonZero(*diffFrame);
  return 100.0 * nonZeroPixels / (diffFrame->rows * diffFrame->cols);
}

void generateBlankFrameAt1Fps(Mat &currFrame, const Size &actualFrameSize) {
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
    currFrame = Mat(actualFrameSize.height, actualFrameSize.width, CV_8UC3,
                    Scalar(128, 128, 128));
  } else {
    currFrame = Mat(540, 960, CV_8UC3, Scalar(128, 128, 128));
    // 960x540, 1280x760, 1920x1080 all have 16:9 aspect ratio.
  }
}
} // namespace FrameHandler
