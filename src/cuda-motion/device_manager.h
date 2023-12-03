#ifndef CM_DEVICE_MANAGER_H
#define CM_DEVICE_MANAGER_H

#include "event_loop.h"
#include "frame_handler.h"
#include "global_vars.h"
#include "ipc.h"
#include "pc_queue.h"
#include "utils.h"

#include <nlohmann/json.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudacodec.hpp>
#include <readerwriterqueue/readerwriterqueue.h>

#include <atomic>
#include <linux/stat.h>
#include <queue>
#include <signal.h>
#include <string>
#include <sys/time.h>

using namespace cv;
using njson = nlohmann::json;

struct videoWritingInfo {
  std::string evaluatedVideoPath;
  float fps;
  ssize_t outputWidth;
  ssize_t outputHeight;
  std::atomic<bool> &videoWriting;
};

struct videoWritingPayload {
  cv::cuda::GpuMat m;
  Ptr<cudacodec::VideoWriter> vw;
};

class DeviceManager : public EventLoop {

public:
  DeviceManager(const size_t deviceIndex);
  ~DeviceManager();
  void getLiveImage(std::vector<uint8_t> &pl);
  std::string getDeviceName() { return this->deviceName; }

protected:
  void InternalThreadEntry();

private:
  std::unique_ptr<IPC> ipc = nullptr;
  njson conf;
  size_t deviceIndex = 0;
  std::string deviceName;
  PercentileTracker<int64_t> pt;

  // frame variables
  bool textOverlayEnabled;
  double textOverlayFontSacle;
  double frameRotationAngle;
  ssize_t outputWidth;
  ssize_t outputHeight;
  std::string evaluatedVideoPath;
  std::unique_ptr<FrameHandler::FrameHandler> fh;

  // motionDetection variables
  enum MotionDetectionMode motionDetectionMode;
  double frameDiffPercentageUpperLimit = 0;
  double frameDiffPercentageLowerLimit = 0;
  double pixelDiffAbsThreshold = 0;
  uint64_t diffEveryNthFrame = 1;
  bool drawContours;

  // videoRecording variables
  uint32_t maxFramesPerVideo;
  uint32_t minFramesPerVideo;
  size_t frameQueueSize;

  // snapshot variables
  int snapshotFrameInterval;

  // mutexes
  std::mutex mtxOnVideoStarts;
  std::mutex mtxOnVideoEnds;
  std::mutex mtxOnDeviceOffline;
  std::mutex mtxOnDeviceBackOnline;

  std::string timestampOnVideoStarts;
  std::string timestampOnDeviceOffline;
  // moodycamel::ReaderWriterQueue<uint64_t> frameTimestamps;
  std::deque<uint64_t> frameTimestamps;

  // videoWritingPcQueue
  PcQueue<cv::cuda::GpuMat, struct videoWritingInfo, struct videoWritingPayload>
      vwPcQueue;
  std::atomic<bool> videoWriting;

  void setParameters(const size_t deviceIndex);
  void
  updateVideoCooldownAndVideoFrameCount(int64_t &cooldown,
                                        uint32_t &videoFrameCount) noexcept;
  bool shouldFrameBeThrottled();
  std::string evaluateStaticVariables(std::basic_string<char> originalString);
  std::string
  evaluateVideoSpecficVariables(std::basic_string<char> originalString);
  void startOrKeepVideoRecording(int64_t &cd);
  void stopVideoRecording(uint32_t &videoFrameCount, int cd);
  void markDeviceAsOffline(bool &isShowingBlankFrame);
  void deviceIsBackOnline(size_t &openRetryDelay, bool &isShowingBlankFrame);
  bool initializeDevice(cudacodec::VideoReader *reader);
  float getCurrentFps();
};

extern std::vector<std::unique_ptr<DeviceManager>> myDevices;

#endif /* CM_DEVICE_MANAGER_H */
