#include "device_manager.h"
#include "frame_handler.h"

#include <spdlog/spdlog.h>

#include <arpa/inet.h>
#include <chrono>
#include <ctime>
#include <errno.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits.h>
#include <regex>
#include <sstream>
#include <sys/mman.h>
#include <sys/socket.h>

using namespace std;

namespace FH = FrameHandler;

deviceManager::deviceManager(const size_t deviceIndex, const njson &defaultConf,
                             njson &overrideConf)
    : zmqContext(1), zmqSocket(zmqContext, zmq::socket_type::pub) {

  setParameters(deviceIndex, defaultConf, overrideConf);

  if (snapshotIpcZeroMQEnabled) {
    /* zmqSocket.bind() throws exception, so we want it to be called
    before other POSIX APIs. */
    try {
      zmqSocket.bind(zeroMQEndpoint);
    } catch (const std::exception &e) {
      spdlog::error("zmqSocket.bind(zeroMQEndpoint): {}", e.what());
      abort();
    }
  }

  if (snapshotIpcHttpEnabled) {
    if (pthread_mutex_init(&mutexLiveImage, NULL) != 0) {
      spdlog::error("pthread_mutex_init() failed, {}({})", errno,
                    strerror(errno));
      goto err_pthread_mutex_init;
    }
  }
  if (snapshotIpcSharedMemEnabled) {
    // Should have used multiple RAII classes to handle this but...
    int shmFd = shm_open(conf["snapshot"]["ipc"]["sharedMem"]["sharedMemName"]
                             .get<string>()
                             .c_str(),
                         O_RDWR | O_CREAT, PERMS);
    if (shmFd < 0) {
      spdlog::error("shm_open() failed, {}({})", errno, strerror(errno));
      goto err_shmFd;
    }
    sharedMemSize = conf["snapshot"]["ipc"]["sharedMem"]["sharedMemSize"];
    if (ftruncate(shmFd, sharedMemSize) != 0) {
      spdlog::error("ftruncate() failed, {}({})", errno, strerror(errno));
      goto err_ftruncate;
    }
    memPtr =
        mmap(NULL, sharedMemSize, PROT_READ | PROT_WRITE, MAP_SHARED, shmFd, 0);
    if (memPtr == MAP_FAILED) {
      spdlog::error("mmap() failed, {}({})", errno, strerror(errno));
      goto err_mmap;
    }
    // umask() is needed to set the correct permissions.
    mode_t old_umask = umask(0);
    semPtr = sem_open(conf["snapshot"]["ipc"]["sharedMem"]["semaphoreName"]
                          .get<string>()
                          .c_str(),
                      O_CREAT | O_RDWR, PERMS, SEM_INITIAL_VALUE);
    umask(old_umask);

    if (semPtr == SEM_FAILED) {
      spdlog::error("sem_open() failed, {}({})", errno, strerror(errno));
      goto err_sem_open;
    }
  }

  return;
err_sem_open:
  if (munmap(memPtr, sharedMemSize) != 0) {
    spdlog::error("munmap() failed: {}({}), "
                  "but there is nothing we can do",
                  errno, strerror(errno));
  }
err_mmap:
err_ftruncate:
  if (shm_unlink(sharedMemName.c_str()) != 0)
    spdlog::error("shm_unlink() failed: {}({}), "
                  "but there is nothing we can do",
                  errno, strerror(errno));
  if (close(shmFd) != 0)
    spdlog::error("close() failed: {}({}), "
                  "but there is nothing we can do",
                  errno, strerror(errno));
err_shmFd:
err_pthread_mutex_init:
  if (snapshotIpcHttpEnabled)
    if (pthread_mutex_destroy(&mutexLiveImage) != 0) {
      spdlog::error("pthread_mutex_destroy() failed:  {}({}), "
                    "but there is nothing we can do",
                    errno, strerror(errno));
    }
  throw runtime_error("initializing deviceManager instance failed. "
                      "Check log for details");
}

string deviceManager::evaluateVideoSpecficVariables(
    basic_string<char> originalString) {
  string filledString =
      regex_replace(originalString, regex(R"(\{\{timestampOnVideoStarts\}\})"),
                    timestampOnVideoStarts);
  filledString =
      regex_replace(filledString, regex(R"(\{\{timestampOnDeviceOffline\}\})"),
                    timestampOnDeviceOffline);
  filledString = regex_replace(filledString, regex(R"(\{\{timestamp\}\})"),
                               getCurrentTimestamp());
  return filledString;
}

string
deviceManager::evaluateStaticVariables(basic_string<char> originalString) {
  string filledString = regex_replace(
      originalString, regex(R"(\{\{deviceIndex\}\})"), to_string(deviceIndex));
  filledString = regex_replace(filledString, regex(R"(\{\{deviceName\}\})"),
                               conf["name"].get<string>());
  return filledString;
}

void deviceManager::setParameters(const size_t deviceIndex,
                                  const njson &defaultConf,
                                  njson &overrideConf) {

  // Most config items will be directly used from njson object, however,
  // given performance concern, for items that are on the critical path,
  // we will duplicate them as class member variables
  this->deviceIndex = deviceIndex;
  conf = overrideConf;
  if (!conf.contains("/name"_json_pointer))
    conf["name"] = defaultConf["name"];
  conf["name"] = evaluateStaticVariables(conf["name"]);
  deviceName = conf["name"];
  if (!conf.contains("/videoFeed/uri"_json_pointer))
    conf["videoFeed"]["uri"] = defaultConf["videoFeed"]["uri"];
  conf["videoFeed"]["uri"] = evaluateStaticVariables(conf["videoFeed"]["uri"]);
  if (!conf.contains("/videoFeed/fourcc"_json_pointer))
    conf["videoFeed"]["fourcc"] = defaultConf["videoFeed"]["fourcc"];
  if (!conf.contains("/videoFeed/videoCaptureApi"_json_pointer))
    conf["videoFeed"]["videoCaptureApi"] =
        defaultConf["videoFeed"]["videoCaptureApi"];

  // ===== frame =====
  if (!conf.contains("/frame/rotation"_json_pointer))
    conf["frame"]["rotation"] = defaultConf["frame"]["rotation"];
  frameRotation = conf["frame"]["rotation"];
  if (!conf.contains("/frame/preferredInputWidth"_json_pointer))
    conf["frame"]["preferredInputWidth"] =
        defaultConf["frame"]["preferredInputWidth"];
  if (!conf.contains("/frame/preferredInputHeight"_json_pointer))
    conf["frame"]["preferredInputHeight"] =
        defaultConf["frame"]["preferredInputHeight"];
  if (!conf.contains("/frame/preferredFps"_json_pointer))
    conf["frame"]["preferredFps"] = defaultConf["frame"]["preferredFps"];

  if (!conf.contains("/frame/outputWidth"_json_pointer))
    conf["frame"]["outputWidth"] = defaultConf["frame"]["outputWidth"];
  outputWidth = conf["frame"]["outputWidth"];
  if (!conf.contains("/frame/outputHeight"_json_pointer))
    conf["frame"]["outputHeight"] = defaultConf["frame"]["outputHeight"];
  outputHeight = conf["frame"]["outputHeight"];

  if (!conf.contains("/frame/throttleFpsIfHigherThan"_json_pointer))
    conf["frame"]["throttleFpsIfHigherThan"] =
        defaultConf["frame"]["throttleFpsIfHigherThan"];
  throttleFpsIfHigherThan = conf["frame"]["throttleFpsIfHigherThan"];
  if (!conf.contains("/frame/textOverlay/enabled"_json_pointer))
    conf["frame"]["textOverlay"]["enabled"] =
        defaultConf["frame"]["textOverlay"]["enabled"];
  textOverlayEnabled = conf["frame"]["textOverlay"]["enabled"];
  if (!conf.contains("/frame/textOverlay/fontScale"_json_pointer))
    conf["frame"]["textOverlay"]["fontScale"] =
        defaultConf["frame"]["textOverlay"]["fontScale"];
  textOverlayFontSacle = conf["frame"]["textOverlay"]["fontScale"];
  if (!conf.contains("/frame/queueSize"_json_pointer))
    conf["frame"]["queueSize"] = defaultConf["frame"]["queueSize"];
  frameQueueSize = conf["frame"]["queueSize"];

  // =====  snapshot =====
  if (!conf.contains("/snapshot/frameInterval"_json_pointer)) {
    conf["snapshot"]["frameInterval"] =
        defaultConf["snapshot"]["frameInterval"];
  }
  snapshotFrameInterval = conf["snapshot"]["frameInterval"];
  if (!conf.contains("/snapshot/ipc/switch/http"_json_pointer)) {
    conf["snapshot"]["ipc"]["switch"]["http"] =
        defaultConf["snapshot"]["ipc"]["switch"]["http"];
  }
  snapshotIpcHttpEnabled = conf["snapshot"]["ipc"]["switch"]["http"];
  if (!conf.contains("/snapshot/ipc/switch/file"_json_pointer)) {
    conf["snapshot"]["ipc"]["switch"]["file"] =
        defaultConf["snapshot"]["ipc"]["switch"]["file"];
  }

  snapshotIpcFileEnabled = conf["snapshot"]["ipc"]["switch"]["file"];
  if (snapshotIpcFileEnabled) {
    if (!conf.contains("/snapshot/ipc/file/path"_json_pointer)) {
      conf["snapshot"]["ipc"]["file"]["path"] =
          defaultConf["snapshot"]["ipc"]["file"]["path"];
    }
    conf["snapshot"]["ipc"]["file"]["path"] =
        evaluateStaticVariables(conf["snapshot"]["ipc"]["file"]["path"]);
    snapshotIpcFilePath = conf["snapshot"]["ipc"]["file"]["path"];
  }

  if (!conf.contains("/snapshot/ipc/switch/sharedMem"_json_pointer)) {
    conf["snapshot"]["ipc"]["switch"]["sharedMem"] =
        defaultConf["snapshot"]["ipc"]["switch"]["sharedMem"];
  }
  snapshotIpcSharedMemEnabled = conf["snapshot"]["ipc"]["switch"]["sharedMem"];
  if (snapshotIpcSharedMemEnabled) {
    if (!conf.contains("/snapshot/ipc/sharedMem/semaphoreName"_json_pointer)) {
      conf["snapshot"]["ipc"]["sharedMem"]["semaphoreName"] =
          defaultConf["snapshot"]["ipc"]["sharedMem"]["semaphoreName"];
    }
    conf["snapshot"]["ipc"]["sharedMem"]["semaphoreName"] =
        evaluateStaticVariables(
            conf["snapshot"]["ipc"]["sharedMem"]["semaphoreName"]);
    if (!conf.contains("/snapshot/ipc/sharedMem/sharedMemName"_json_pointer)) {
      conf["snapshot"]["ipc"]["sharedMem"]["sharedMemName"] =
          defaultConf["snapshot"]["ipc"]["sharedMem"]["sharedMemName"];
    }
    conf["snapshot"]["ipc"]["sharedMem"]["sharedMemName"] =
        evaluateStaticVariables(
            conf["snapshot"]["ipc"]["sharedMem"]["sharedMemName"]);
    if (!conf.contains("/snapshot/ipc/sharedMem/sharedMemSize"_json_pointer)) {
      conf["snapshot"]["ipc"]["sharedMem"]["sharedMemSize"] =
          defaultConf["snapshot"]["ipc"]["sharedMem"]["sharedMemSize"];
    }
  }

  if (!conf.contains("/snapshot/ipc/switch/zeroMQ"_json_pointer)) {
    conf["snapshot"]["ipc"]["switch"]["zeroMQ"] =
        defaultConf["snapshot"]["ipc"]["switch"]["zeroMQ"];
  }
  snapshotIpcZeroMQEnabled = conf["snapshot"]["ipc"]["switch"]["zeroMQ"];
  if (snapshotIpcZeroMQEnabled) {
    if (!conf.contains("/snapshot/ipc/zeroMQ/endpoint"_json_pointer)) {
      conf["snapshot"]["ipc"]["zeroMQ"]["endpoint"] =
          defaultConf["snapshot"]["ipc"]["zeroMQ"]["endpoint"];
    }
    conf["snapshot"]["ipc"]["zeroMQ"]["endpoint"] =
        evaluateStaticVariables(conf["snapshot"]["ipc"]["zeroMQ"]["endpoint"]);
    zeroMQEndpoint = conf["snapshot"]["ipc"]["zeroMQ"]["endpoint"];
  }

  // ===== events =====
  if (!conf.contains("/events/onVideoStarts"_json_pointer))
    conf["events"]["onVideoStarts"] = defaultConf["events"]["onVideoStarts"];
  for (size_t i = 0; i < conf["events"]["onVideoStarts"].size(); ++i) {
    conf["events"]["onVideoStarts"][i] =
        evaluateStaticVariables(conf["events"]["onVideoStarts"][i]);
  }
  if (!conf.contains("/events/onVideoEnds"_json_pointer))
    conf["events"]["onVideoEnds"] = defaultConf["events"]["onVideoEnds"];
  for (size_t i = 0; i < conf["events"]["onVideoEnds"].size(); ++i) {
    conf["events"]["onVideoEnds"][i] =
        evaluateStaticVariables(conf["events"]["onVideoEnds"][i]);
  }
  if (!conf.contains("/events/onDeviceOffline"_json_pointer))
    conf["events"]["onDeviceOffline"] =
        defaultConf["events"]["onDeviceOffline"];
  for (size_t i = 0; i < conf["events"]["onDeviceOffline"].size(); ++i) {
    conf["events"]["onDeviceOffline"][i] =
        evaluateStaticVariables(conf["events"]["onDeviceOffline"][i]);
  }
  if (!conf.contains("/events/onDeviceBackOnline"_json_pointer))
    conf["events"]["onDeviceBackOnline"] =
        defaultConf["events"]["onDeviceBackOnline"];
  for (size_t i = 0; i < conf["events"]["onDeviceBackOnline"].size(); ++i) {
    conf["events"]["onDeviceBackOnline"][i] =
        evaluateStaticVariables(conf["events"]["onDeviceBackOnline"][i]);
  }

  // ===== motion detection =====
  if (!conf.contains("/motionDetection/mode"_json_pointer))
    conf["motionDetection"]["mode"] = defaultConf["motionDetection"]["mode"];
  motionDetectionMode = conf["motionDetection"]["mode"];
  if (!conf.contains(
          "/motionDetection/frameDiffPercentageLowerLimit"_json_pointer))
    conf["motionDetection"]["frameDiffPercentageLowerLimit"] =
        defaultConf["motionDetection"]["frameDiffPercentageLowerLimit"];
  frameDiffPercentageLowerLimit =
      conf["motionDetection"]["frameDiffPercentageLowerLimit"];
  if (!conf.contains(
          "/motionDetection/frameDiffPercentageUpperLimit"_json_pointer))
    conf["motionDetection"]["frameDiffPercentageUpperLimit"] =
        defaultConf["motionDetection"]["frameDiffPercentageUpperLimit"];
  frameDiffPercentageUpperLimit =
      conf["motionDetection"]["frameDiffPercentageUpperLimit"];
  if (!conf.contains("/motionDetection/pixelDiffAbsThreshold"_json_pointer))
    conf["motionDetection"]["pixelDiffAbsThreshold"] =
        defaultConf["motionDetection"]["pixelDiffAbsThreshold"];
  pixelDiffAbsThreshold = conf["motionDetection"]["pixelDiffAbsThreshold"];
  if (!conf.contains("/motionDetection/diffEveryNthFrame"_json_pointer))
    conf["motionDetection"]["diffEveryNthFrame"] =
        defaultConf["motionDetection"]["diffEveryNthFrame"];
  diffEveryNthFrame = conf["motionDetection"]["diffEveryNthFrame"];
  if (diffEveryNthFrame == 0) {
    throw invalid_argument("diffEveryNthFrame must be greater than 0");
  }
  if (!conf.contains(
          "/motionDetection/videoRecording/minFramesPerVideo"_json_pointer))
    conf["motionDetection"]["videoRecording"]["minFramesPerVideo"] =
        defaultConf["motionDetection"]["videoRecording"]["minFramesPerVideo"];
  minFramesPerVideo =
      conf["motionDetection"]["videoRecording"]["minFramesPerVideo"];
  if (!conf.contains(
          "/motionDetection/videoRecording/maxFramesPerVideo"_json_pointer))
    conf["motionDetection"]["videoRecording"]["maxFramesPerVideo"] =
        defaultConf["motionDetection"]["videoRecording"]["maxFramesPerVideo"];
  maxFramesPerVideo =
      conf["motionDetection"]["videoRecording"]["maxFramesPerVideo"];

  if (!conf.contains(
          "/motionDetection/videoRecording/videoWriter/fourcc"_json_pointer))
    conf["motionDetection"]["videoRecording"]["videoWriter"]["fourcc"] =
        defaultConf["motionDetection"]["videoRecording"]["videoWriter"]
                   ["fourcc"];
  if (!conf.contains(
          "/motionDetection/videoRecording/videoWriter/videoPath"_json_pointer))
    conf["motionDetection"]["videoRecording"]["videoWriter"]["videoPath"] =
        defaultConf["motionDetection"]["videoRecording"]["videoWriter"]
                   ["videoPath"];
  conf["motionDetection"]["videoRecording"]["videoWriter"]["videoPath"] =
      evaluateStaticVariables(conf["motionDetection"]["videoRecording"]
                                  ["videoWriter"]["videoPath"]);
  if (!conf.contains(
          "/motionDetection/videoRecording/videoWriter/fps"_json_pointer))
    conf["motionDetection"]["videoRecording"]["videoWriter"]["fps"] =
        defaultConf["motionDetection"]["videoRecording"]["videoWriter"]["fps"];

  if (!conf.contains("/motionDetection/drawContours"_json_pointer))
    conf["motionDetection"]["drawContours"] =
        defaultConf["motionDetection"]["drawContours"];
  drawContours = conf["motionDetection"]["drawContours"];

  spdlog::info("{}-th device to be used with the following configs:\n{}",
               deviceIndex, conf.dump(2));
}

deviceManager::~deviceManager() {
  if (snapshotIpcHttpEnabled) {
    if (pthread_mutex_destroy(&mutexLiveImage) != 0) {
      spdlog::error("pthread_mutex_destroy() failed: {}, "
                    "but there is nothing we can do",
                    errno);
    }
  }
  if (snapshotIpcSharedMemEnabled) {
    // unlink and close() are both needed: unlink only disassociates the
    // name from the underlying semaphore object, but the semaphore object
    // is not gone. It will only be gone when we close() it.
    if (sem_unlink(semaphoreName.c_str()) != 0) {
      spdlog::error("sem_unlink() failed: {}({}), "
                    "but there is nothing we can do",
                    errno, strerror(errno));
    }
    if (sem_close(semPtr) != 0) {
      spdlog::error("sem_close() failed: {}({}), "
                    "but there is nothing we can do",
                    errno, strerror(errno));
    }
    if (munmap(memPtr, sharedMemSize) != 0) {
      spdlog::error("munmap() failed: {}({}), "
                    "but there is nothing we can do",
                    errno, strerror(errno));
    }
    if (shm_unlink(sharedMemName.c_str()) != 0) {
      spdlog::error("shm_unlink() failed: {}({}), "
                    "but there is nothing we can do",
                    errno, strerror(errno));
    }
    if (close(shmFd) != 0) {
      spdlog::error("close() failed: {}({}), "
                    "but there is nothing we can do",
                    errno, strerror(errno));
    }
  }
}

float deviceManager::getCurrentFps(int64_t msSinceEpoch) {
  float fps = FLT_MAX;
  if (msSinceEpoch - frameTimestamps.front() > 0) {
    fps = 1000.0 * frameTimestamps.size() /
          (msSinceEpoch - frameTimestamps.front());
  }
  return fps;
}

bool deviceManager::shouldFrameBeThrottled() {

  constexpr int sampleMsUpperLimit = 10 * 1000;
  int64_t msSinceEpoch = chrono::duration_cast<chrono::milliseconds>(
                             chrono::system_clock::now().time_since_epoch())
                             .count();
  if (frameTimestamps.size() <= 1) {
    frameTimestamps.push(msSinceEpoch);
    return false;
  }

  // time complexity of frameTimestamps.size()/front()/back()/push()/pop()
  // are guaranteed to be O(1)
  if (msSinceEpoch - frameTimestamps.front() > sampleMsUpperLimit) {
    frameTimestamps.pop();
  }
  // The logic is this:
  // A timestamp is only added to the queue if it is not throttled.
  // That is, if a frame is throttled, it will not be taking into account
  // when we calculate Fps.
  if (getCurrentFps(msSinceEpoch) > throttleFpsIfHigherThan) {
    return true;
  }
  frameTimestamps.push(msSinceEpoch);
  return false;
}

void deviceManager::asyncExecCallback(void *This, string stdout, string stderr,
                                      int rc) {
  if (stdout.size() > 0) {
    spdlog::info("[{}] non-empty stdout from command: [{}]",
                 reinterpret_cast<deviceManager *>(This)->deviceName, stdout);
  }
  if (stderr.size() > 0) {
    spdlog::info("[{}] non-empty stderr from command: [{}]",
                 reinterpret_cast<deviceManager *>(This)->deviceName, stderr,
                 rc);
  }
  if (rc != 0) {
    spdlog::info("[{}] non-zero return code from command: [{}]",
                 reinterpret_cast<deviceManager *>(This)->deviceName, rc);
  }
}

void deviceManager::stopVideoRecording(VideoWriter &vwriter,
                                       uint32_t &videoFrameCount, int cd) {

  vwriter.release();

  if (cd > 0) {
    spdlog::warn("[{}] video recording stopped before cooldown "
                 "reaches 0",
                 deviceName);
  }

  if (conf["events"]["onVideoEnds"].size() > 0 && cd == 0) {
    vector<string> args;
    args.reserve(conf["events"]["onVideoEnds"].size());
    spdlog::info("[{}] video recording ends", deviceName);
    for (size_t i = 0; i < conf["events"]["onVideoEnds"].size(); ++i) {
      args.push_back(
          evaluateVideoSpecficVariables(conf["events"]["onVideoEnds"][i]));
    }
    execAsync((void *)this, args, asyncExecCallback);
    spdlog::info("[{}] onVideoEnds triggered, command [{}] executed",
                 deviceName, args[0]);
  } else if (conf["events"]["onVideoEnds"].size() > 0 && cd > 0) {
    spdlog::warn("[{}] onVideoEnds event defined but it won't be "
                 "triggered",
                 deviceName);
  } else {
    spdlog::info("[{}] onVideoEnds, no command to execute", deviceName);
  }
  videoFrameCount = 0;
}

void deviceManager::startOrKeepVideoRecording(VideoWriter &vwriter,
                                              int64_t &cd) {

  auto handleOnVideoStarts = [&]() {
    if (conf["events"]["onVideoStarts"].size() > 0) {
      vector<string> args;
      args.reserve(conf["events"]["onVideoStarts"].size());
      for (size_t i = 0; i < conf["events"]["onVideoStarts"].size(); ++i) {
        args.push_back(
            evaluateVideoSpecficVariables(conf["events"]["onVideoStarts"][i]));
      }
      spdlog::info("[{}] motion detected, video recording begins", deviceName);
      execAsync((void *)this, args, asyncExecCallback);
      spdlog::info("[{}] onVideoStarts: command [{}] executed", deviceName,
                   args[0]);
    } else {
      spdlog::info("[{}] onVideoStarts: no command to execute", deviceName);
    }
  };

  cd = minFramesPerVideo;

  // These two if's: video recording is in progress already.
  if (vwriter.isOpened())
    return;

  timestampOnVideoStarts = getCurrentTimestamp();
  evaluatedVideoPath = evaluateVideoSpecficVariables(
      conf["motionDetection"]["videoRecording"]["videoWriter"]["videoPath"]);

  handleOnVideoStarts();
  // Use OpenCV to encode video
  string fourcc =
      conf["motionDetection"]["videoRecording"]["videoWriter"]["fourcc"]
          .get<string>();
  const int codec =
      VideoWriter::fourcc(fourcc[0], fourcc[1], fourcc[2], fourcc[3]);
  /* For VideoWriter, we have to use FFmpeg as we compiled FFmpeg with
  Nvidia GPU*/
  vwriter = VideoWriter(
      evaluatedVideoPath, cv::CAP_FFMPEG, codec,
      conf["motionDetection"]["videoRecording"]["videoWriter"]["fps"],
      Size(outputWidth, outputHeight),
      {VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY});
}

void deviceManager::getLiveImage(vector<uint8_t> &pl) {
  if (encodedJpgImage.size() > 0) {
    if (pthread_mutex_lock(&mutexLiveImage) != 0) {
      spdlog::error("[{}] pthread_mutex_lock() failed, "
                    "will skip live image data copy, errno: {}",
                    deviceName, errno);
      return;
    }
    pl = encodedJpgImage;
    if (pthread_mutex_unlock(&mutexLiveImage) != 0) {
      spdlog::error("[{}] pthread_mutex_unlock() failed, errno: {}", deviceName,
                    errno);
      return;
    }
  } else {
    pl = vector<uint8_t>();
  }
}

void deviceManager::generateBlankFrameAt1Fps(Mat &currFrame,
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
    currFrame = Mat(actualFrameSize.height, actualFrameSize.width, CV_8UC3,
                    Scalar(128, 128, 128));
  } else {
    currFrame = Mat(540, 960, CV_8UC3, Scalar(128, 128, 128));
    // 960x540, 1280x760, 1920x1080 all have 16:9 aspect ratio.
  }
}

void deviceManager::updateVideoCooldownAndVideoFrameCount(
    int64_t &cd, uint32_t &videoFrameCount) {
  if (cd >= 0) {
    cd--;
    if (cd > 0) {
      ++videoFrameCount;
    }
  }
  if (videoFrameCount >= maxFramesPerVideo) {
    cd = 0;
  }
}

void deviceManager::markDeviceAsOffline(bool &isShowingBlankFrame) {

  if (isShowingBlankFrame == false) {
    timestampOnDeviceOffline = getCurrentTimestamp();
    isShowingBlankFrame = true;
    if (conf["events"]["onDeviceOffline"].size() > 0) {
      vector<string> args;
      args.reserve(conf["events"]["onDeviceOffline"].size());
      for (size_t i = 0; i < conf["events"]["onDeviceOffline"].size(); ++i) {
        args.push_back(evaluateVideoSpecficVariables(
            conf["events"]["onDeviceOffline"][i]));
      }
      execAsync((void *)this, args, asyncExecCallback);
      spdlog::info("[{}] onDeviceOffline: command [{}] executed", deviceName,
                   args[0]);
    } else {
      spdlog::info("[{}] onDeviceOffline: no command to execute", deviceName);
    }
  }
}

void deviceManager::deviceIsBackOnline(size_t &openRetryDelay,
                                       bool &isShowingBlankFrame) {
  spdlog::info("[{}] Device is back online", deviceName);
  timestampOnDeviceOffline = "";
  openRetryDelay = 1;
  isShowingBlankFrame = false;
  if (conf["events"]["onDeviceBackOnline"].size() > 0) {
    vector<string> args;
    args.reserve(conf["events"]["onDeviceBackOnline"].size());
    for (size_t i = 0; i < conf["events"]["onDeviceBackOnline"].size(); ++i) {
      args.push_back(evaluateVideoSpecficVariables(
          conf["events"]["onDeviceBackOnline"][i]));
    }
    execAsync((void *)this, args, asyncExecCallback);
    spdlog::info("[{}] onDeviceBackOnline: command [{}] executed", deviceName,
                 args[0]);
  }
}

void deviceManager::initializeDevice(VideoCapture &cap, bool &result,
                                     const Size &actualFrameSize) {
  result = cap.open(conf["videoFeed"]["uri"].get<string>(),
                    conf["videoFeed"]["videoCaptureApi"],
                    {CAP_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY});
  spdlog::info("[{}] cap.open({}): {}", deviceName,
               conf["videoFeed"]["uri"].get<string>(), result);
  if (!result) {
    return;
  }

  int fourcc = cap.get(CAP_PROP_FOURCC);
  string fourcc_str = format("%c%c%c%c", fourcc & 255, (fourcc >> 8) & 255,
                             (fourcc >> 16) & 255, (fourcc >> 24) & 255);

  if (all_of(fourcc_str.begin(), fourcc_str.end(),
             [](char c) { return !isgraph(c); })) {
    fourcc_str = "<non-printable>";
  }
  spdlog::info("[{}] cap.getBackendName(): {}, CAP_PROP_FOURCC: 0x{:x}({})",
               deviceName, cap.getBackendName(), fourcc, fourcc_str);
  string fcc = conf["videoFeed"]["fourcc"];
  if (fcc.size() == 4) {
    cap.set(CAP_PROP_FOURCC,
            VideoWriter::fourcc(fcc[0], fcc[1], fcc[2], fcc[3]));
  }
  if (actualFrameSize.width > 0)
    cap.set(CAP_PROP_FRAME_WIDTH, actualFrameSize.width);
  if (actualFrameSize.height > 0)
    cap.set(CAP_PROP_FRAME_HEIGHT, actualFrameSize.height);
  if (conf["frame"]["preferredFps"] > 0)
    cap.set(CAP_PROP_FPS, conf["frame"]["preferredFps"]);
}

void deviceManager::warnCPUResize(const Size &actualFrameSize) {
  if ((actualFrameSize.width != conf["frame"]["preferredInputWidth"] &&
       conf["frame"]["preferredInputWidth"] != -1) ||
      (actualFrameSize.height != conf["frame"]["preferredInputHeight"] &&
       conf["frame"]["preferredInputHeight"] != -1)) {
    spdlog::warn("[{}] actualFrameSize({}x{}) is different "
                 "from preferredInputSize ({}x{}). (The program has "
                 "no choice but using actualFrameSize)",
                 deviceName, actualFrameSize.width, actualFrameSize.height,
                 conf["frame"]["preferredInputWidth"].get<int>(),
                 conf["frame"]["preferredInputHeight"].get<int>());
  }
  if (actualFrameSize.width != outputWidth ||
      actualFrameSize.height != outputHeight) {
    spdlog::warn("[{}] actualFrameSize({}x{}) is different "
                 "from outputSize ({}x{}), so frame will have to be "
                 "resized, this is a CPU-intensive operation.",
                 deviceName, actualFrameSize.width, actualFrameSize.height,
                 outputWidth, outputHeight);
  }
}

void deviceManager::prepareDataForIpc(Mat &dispFrame) {
  // vector<int> configs = {IMWRITE_JPEG_QUALITY, 80};
  vector<int> configs = {};
  if (snapshotIpcHttpEnabled) {
    pthread_mutex_lock(&mutexLiveImage);
    imencode(".jpg", dispFrame, encodedJpgImage, configs);
    pthread_mutex_unlock(&mutexLiveImage);
  } else {
    imencode(".jpg", dispFrame, encodedJpgImage, configs);
  }

  // Profiling show that the above mutex section without actual
  // waiting takes ~30 ms to complete, means that the CPU can only
  // handle ~30 fps

  // https://stackoverflow.com/questions/7054844/is-rename-atomic
  // https://stackoverflow.com/questions/29261648/atomic-writing-to-file-on-linux
  if (snapshotIpcFileEnabled) {
    string sp = evaluateVideoSpecficVariables(snapshotIpcFilePath);
    ofstream fout(sp + ".tmp", ios::out | ios::binary);
    if (!fout.good()) {
      spdlog::error("[{}] Failed to open file [{}]: {}", deviceName,
                    sp + ".tmp", strerror(errno));
    } else {
      fout.write((char *)encodedJpgImage.data(), encodedJpgImage.size());
      if (!fout.good()) {
        spdlog::error("[{}] Failed to write to file [{}]: {}", deviceName,
                      sp + ".tmp", strerror(errno));
      }
      fout.close();
      if (rename((sp + ".tmp").c_str(), sp.c_str()) != 0) {
        spdlog::error("[{}] Failed to rename [{}] tp [{}]: {}", deviceName,
                      sp + ".tmp", sp, strerror(errno));
      }
    }
    // profiling shows from ofstream fout()... to rename() takes
    // less than 1 ms.
  }

  if (snapshotIpcSharedMemEnabled) {
    size_t s = encodedJpgImage.size();
    if (s > sharedMemSize - sizeof(size_t)) {
      spdlog::error("[{}] encodedJpgImage({} bytes) too large for "
                    "sharedMemSize({} bytes)",
                    deviceName, s, sharedMemSize);
      s = 0;
    } else {
      if (sem_wait(semPtr) != 0) {
        spdlog::error(
            "[{}] sem_wait() failed: {}, the program will "
            "continue to run but the semaphore mechanism could be broken",
            deviceName, errno);
      } else {

        memcpy(memPtr, &s, sizeof(size_t));
        memcpy((uint8_t *)memPtr + sizeof(size_t), encodedJpgImage.data(),
               encodedJpgImage.size());
        if (sem_post(semPtr) != 0) {
          spdlog::error(
              "[{}] sem_post() failed: {}, the program will "
              "continue to run but the semaphore mechanism could be broken",
              deviceName, errno);
        }
      }
    }
  }

  if (snapshotIpcZeroMQEnabled) {
    zmqSocket.send(encodedJpgImage.data(), encodedJpgImage.size());
  }
}

void deviceManager::InternalThreadEntry() {

  queue<Mat> dispFrames;
  Mat prevFrame, currFrame, diffFrame;
  bool result = false;
  bool isShowingBlankFrame = false;
  VideoCapture cap;
  float rateOfChange = 0.0;

  uint64_t retrievedFramesSinceLastOpen = 0;
  uint64_t retrievedFramesSinceStart = 0;
  uint32_t videoFrameCount = 0;
  Size actualFrameSize = Size(conf["frame"]["preferredInputWidth"],
                              conf["frame"]["preferredInputHeight"]);
  VideoWriter vwriter;
  // cd: cooldown
  int64_t cd = 0;
  size_t openRetryDelay = 1;

  // We use the evil `goto` statement so that we can avoid the duplication of
  //   initializeDevice()...
  goto entryPoint;

  while (ev_flag == 0) {
    result = cap.read(currFrame);
    if (shouldFrameBeThrottled()) {
      /* Seems that sometimes OpenCV could grab() the same frame time
      and time again, maxing out the CPU, so we make it sleep_for() a
      little bit of time. */
      this_thread::sleep_for(2ms);
      continue;
    }
    ++retrievedFramesSinceStart;
    ++retrievedFramesSinceLastOpen;

    if (result == false || currFrame.empty() || cap.isOpened() == false) {
      markDeviceAsOffline(isShowingBlankFrame);
      generateBlankFrameAt1Fps(currFrame, actualFrameSize);
      if (retrievedFramesSinceStart % openRetryDelay == 0) {
        openRetryDelay *= 2;
        spdlog::error("[{}] Unable to cap.read() a new frame. "
                      "Wait for {} frames than then re-open()...",
                      deviceName, openRetryDelay);
      entryPoint:
        initializeDevice(cap, result, actualFrameSize);
        retrievedFramesSinceLastOpen = 0;
        continue;
      }
    } else if (isShowingBlankFrame) {
      deviceIsBackOnline(openRetryDelay, isShowingBlankFrame);
    }
    if (frameRotation != -1 && isShowingBlankFrame == false) {
      rotate(currFrame, currFrame, frameRotation);
    }
    if (retrievedFramesSinceLastOpen == 1) {
      actualFrameSize = currFrame.size();
      outputWidth = outputWidth == -1 ? actualFrameSize.width : outputWidth;
      outputHeight = outputHeight == -1 ? actualFrameSize.height : outputHeight;
      warnCPUResize(actualFrameSize);
    }

    if (motionDetectionMode == MODE_DETECT_MOTION &&
        retrievedFramesSinceStart % diffEveryNthFrame == 0) {
      if (isShowingBlankFrame == false) {
        // profiling shows this if() block takes around 1-2 ms
        rateOfChange = FH::getFrameChanges(prevFrame, currFrame, &diffFrame,
                                           pixelDiffAbsThreshold);
      } else {
        rateOfChange = -1;
      }
      /* Can't just assign like prevFrame = currFrame, otherwise two
      objects will share the same copy of underlying image data */
      prevFrame = currFrame.clone();
    }
    if (actualFrameSize.width != outputWidth ||
        actualFrameSize.height != outputHeight) {
      resize(currFrame, currFrame, cv::Size(outputWidth, outputHeight));
    }
    dispFrames.push(currFrame);
    if (dispFrames.size() > frameQueueSize) {
      dispFrames.pop();
    }
    if (drawContours && motionDetectionMode == MODE_DETECT_MOTION &&
        isShowingBlankFrame == false) {
      FH::overlayContours(dispFrames.back(), diffFrame);
      // CPU-intensive! Use with care!
    }
    if (textOverlayEnabled) {
      FH::overlayStats(
          dispFrames.back(), rateOfChange, cd, videoFrameCount,
          textOverlayFontSacle, motionDetectionMode,
          getCurrentFps(chrono::duration_cast<chrono::milliseconds>(
                            chrono::system_clock::now().time_since_epoch())
                            .count()),
          maxFramesPerVideo);
      FH::overlayDatetime(dispFrames.back(), textOverlayFontSacle,
                          timestampOnDeviceOffline);
      FH::overlayDeviceName(dispFrames.back(), textOverlayFontSacle,
                            deviceName);
    }

    if ((retrievedFramesSinceStart - 1) % snapshotFrameInterval == 0) {
      prepareDataForIpc(dispFrames.front());
    }

    if (motionDetectionMode == MODE_DISABLED) {
      continue;
    }

    if (motionDetectionMode == MODE_ALWAYS_RECORD ||
        (rateOfChange > frameDiffPercentageLowerLimit &&
         rateOfChange < frameDiffPercentageUpperLimit &&
         motionDetectionMode == MODE_DETECT_MOTION)) {
      startOrKeepVideoRecording(vwriter, cd);
    }

    updateVideoCooldownAndVideoFrameCount(cd, videoFrameCount);

    if (cd < 0) {
      continue;
    }
    if (cd == 0) {
      stopVideoRecording(vwriter, videoFrameCount, cd);
      continue;
    }
    if (dispFrames.front().size().width != outputWidth ||
        dispFrames.front().size().height != outputHeight) {
      spdlog::warn("dispFrame.size() == (w{}, h{}) != Size(outputWidth, "
                   "outputHeight) == (w{}, h{}), "
                   "OpenCV::VideoWriter may fail silently",
                   dispFrames.front().size().width,
                   dispFrames.front().size().height, outputWidth, outputHeight);
    }
    vwriter.write(dispFrames.front());
  }
  if (cd > 0) {
    stopVideoRecording(vwriter, videoFrameCount, cd);
  }
  cap.release();
  spdlog::info("[{}] thread quits gracefully", deviceName);
}
