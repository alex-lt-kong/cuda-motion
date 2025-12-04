#include "frame_handler.h"
#include "utils.h"

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <spdlog/spdlog.h>

#include <iomanip>

using namespace cv;
using namespace std;

namespace CudaMotion::Utils {

FrameHandler::FrameHandler(const double fontScale, const string &deviceName)
    : pt(10000) {
  count = 0;
  overlayDatetimeBufLen = UINT_MAX;
  this->fontScale = fontScale;
  this->deviceName = deviceName;
  overlayDeviceNameTextSize = getTextSize(this->deviceName, FONT_HERSHEY_DUPLEX,
                                          fontScale, 8 * fontScale, nullptr);
}


bool FrameHandler::nextDummyFrame(cuda::GpuMat &frame, const Size &frameSize) {

  // assume the video source is at 30fps
  this_thread::sleep_for(chrono::milliseconds(1000 / 30));
  try {
    frame = cuda::GpuMat(frameSize.height, frameSize.width, CV_8UC3,
                         Scalar(128, 128, 128));
    return true;
  } catch (const cv::Exception &e) {
    // This is by no means cosmetic--it does happen when GPU failed to allocate
    // CUDA memory for even one frame
    spdlog::error("[{}] nextDummyFrame() failed: {}", deviceName, e.what());
  }
  return false;
}
} // namespace CudaMotion::Utils
