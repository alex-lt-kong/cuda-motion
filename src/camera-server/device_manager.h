#ifndef CS_DEVICE_MANAGER_H
#define CS_DEVICE_MANAGER_H

#include "event_loop.h"
#include "global_vars.h"
#include "utils.h"

#include <nlohmann/json.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <zmq.hpp>

#include <linux/stat.h>
#include <pthread.h>
#include <queue>
#include <semaphore.h>
#include <signal.h>
#include <string>
#include <sys/time.h>

using namespace cv;
using njson = nlohmann::json;

#define PERMS (S_IRWXU | S_IRWXG | S_IRWXO)
#define SEM_INITIAL_VALUE 1

class deviceManager : public MyEventLoopThread {

public:
  deviceManager(const size_t deviceIndex, const njson &defaultConf,
                njson &overrideConf);
  ~deviceManager();
  void setParameters(const size_t deviceIndex, const njson &defaultConf,
                     njson &overrideConf);
  void getLiveImage(std::vector<uint8_t> &pl);
  std::string getDeviceName() { return this->deviceName; }

protected:
  void InternalThreadEntry();

private:
  pthread_mutex_t mutexLiveImage;
  std::vector<uint8_t> encodedJpgImage;
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
  bool snapshotIpcFileEnabled;
  bool snapshotIpcHttpEnabled;
  std::string snapshotIpcFilePath;
  bool snapshotHttpFileEnabled;
  bool snapshotIpcSharedMemEnabled;
  int shmFd;
  size_t sharedMemSize;
  std::string sharedMemName;
  std::string semaphoreName;
  void *memPtr;
  sem_t *semPtr;
  bool snapshotIpcZeroMQEnabled;
  std::string zeroMQEndpoint;
  zmq::context_t zmqContext;
  zmq::socket_t zmqSocket;

  std::string timestampOnVideoStarts;
  std::string timestampOnDeviceOffline;
  std::queue<int64_t> frameTimestamps;

  volatile sig_atomic_t *done;

  void updateVideoCooldownAndVideoFrameCount(int64_t &cooldown,
                                             uint32_t &videoFrameCount);
  bool shouldFrameBeThrottled();
  std::string evaluateStaticVariables(std::basic_string<char> originalString);
  std::string
  evaluateVideoSpecficVariables(std::basic_string<char> originalString);
  std::string convertToString(char *a, int size);
  void startOrKeepVideoRecording(VideoWriter &vwriter, int64_t &cd);
  void stopVideoRecording(VideoWriter &vwriter, uint32_t &videoFrameCount,
                          int cd);
  void generateBlankFrameAt1Fps(Mat &currFrame, const Size &actualFrameSize);
  void markDeviceAsOffline(bool &isShowingBlankFrame);
  void deviceIsBackOnline(size_t &openRetryDelay, bool &isShowingBlankFrame);
  void initializeDevice(VideoCapture &cap, bool &result,
                        const Size &actualFrameSize);
  static void asyncExecCallback(void *This, std::string stdout,
                                std::string stderr, int rc);
  void prepareDataForIpc(Mat &dispFrames);
  float getCurrentFps(int64_t msSinceEpoch);
  void warnCPUResize(const Size &actualFrameSize);
};

#endif
