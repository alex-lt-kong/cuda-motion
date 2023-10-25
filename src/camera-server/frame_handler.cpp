#include "frame_handler.h"
#include "utils.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iomanip>

using namespace cv;
using namespace std;

namespace FrameHandler {

void overlayDatetime(Mat &frame, double fontSacle,
                     string timestampOnDeviceOffline) {
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

void overlayDeviceName(Mat &frame, double fontSacle, string deviceName) {

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

void overlayStats(Mat &frame, float changeRate, int cd,
                  long long int videoFrameCount, double fontSacle,
                  enum MotionDetectionMode mode, float currentFps,
                  uint32_t maxFramesPerVideo) {
  ostringstream textToOverlay;
  if (mode == MODE_DETECT_MOTION) {
    textToOverlay << fixed << setprecision(2) << changeRate << "%, ";
  }
  textToOverlay << fixed << setprecision(1) << currentFps << "fps ";
  if (mode != MODE_DISABLED) {
    textToOverlay << "(" << cd << ", " << maxFramesPerVideo - videoFrameCount
                  << ")";
  }
  putText(frame, textToOverlay.str(), Point(5, frame.rows - 5),
          FONT_HERSHEY_DUPLEX, fontSacle, Scalar(0, 0, 0), 8 * fontSacle,
          LINE_8, false);
  putText(frame, textToOverlay.str(), Point(5, frame.rows - 5),
          FONT_HERSHEY_DUPLEX, fontSacle, Scalar(255, 255, 255), 2 * fontSacle,
          LINE_8, false);
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

} // namespace FrameHandler