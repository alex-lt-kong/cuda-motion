#ifndef CS_DEVICE_MANAGER_H
#define CS_DEVICE_MANAGER_H

#include "event_loop.h"
#include "global_vars.h"
#include "ipc.h"
#include "utils.h"

#include <nlohmann/json.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <linux/stat.h>
#include <queue>
#include <signal.h>
#include <string>
#include <sys/time.h>

using namespace cv;
using njson = nlohmann::json;

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

  // frame variables
  bool textOverlayEnabled;
  double textOverlayFontSacle;
  float throttleFpsIfHigherThan;
  int frameRotation;
  ssize_t outputWidth;
  ssize_t outputHeight;
  std::string evaluatedVideoPath;

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
  std::queue<int64_t> frameTimestamps;

  void setParameters(const size_t deviceIndex, const njson &defaultConf,
                     njson &overrideConf);
  void updateVideoCooldownAndVideoFrameCount(int64_t &cooldown,
                                             uint32_t &videoFrameCount);
  bool shouldFrameBeThrottled();
  std::string evaluateStaticVariables(std::basic_string<char> originalString);
  std::string
  evaluateVideoSpecficVariables(std::basic_string<char> originalString);
  void startOrKeepVideoRecording(VideoWriter &vwriter, int64_t &cd);
  void stopVideoRecording(VideoWriter &vwriter, uint32_t &videoFrameCount,
                          int cd);
  void markDeviceAsOffline(bool &isShowingBlankFrame);
  void deviceIsBackOnline(size_t &openRetryDelay, bool &isShowingBlankFrame);
  void initializeDevice(VideoCapture &cap, bool &result,
                        const Size &actualFrameSize);
  float getCurrentFps(int64_t msSinceEpoch);
  void warnCPUResize(const Size &actualFrameSize);
};

extern std::vector<std::unique_ptr<DeviceManager>> myDevices;

#endif /* CS_DEVICE_MANAGER_H */
