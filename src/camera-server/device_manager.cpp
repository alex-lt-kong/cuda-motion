#include "device_manager.h"
#include "frame_handler.h"

#include <spdlog/spdlog.h>

#include <errno.h>
#include <filesystem>
#include <limits.h>
#include <regex>
#include <sstream>
#include <sys/socket.h>

using namespace std;

namespace FH = FrameHandler;

DeviceManager::DeviceManager(const size_t deviceIndex) {

  { // settings variable is used by multiple threads, possibly concurrently, is
    // reading it thread-safe? According to the below response from the library
    // author: https://github.com/nlohmann/json/issues/651
    // The answer seems to be affirmative.
    // But writing to settings is unlikely to be thread-safe:
    // https://github.com/nlohmann/json/issues/651
    lock_guard<mutex> guard(mtxNjsonSettings);
    auto t = settings["devices"][deviceIndex];
    settings["devices"][deviceIndex] = settings["devicesDefault"];
    settings["devices"][deviceIndex].merge_patch(t);
    conf = settings["devices"][deviceIndex];
    setParameters(deviceIndex);
  }
  ipc = make_unique<IPC>(deviceIndex, deviceName);

  if (conf.value("/snapshot/ipc/http/enabled"_json_pointer, false)) {
    ipc->enableHttp();
  }
  if (conf.value("/snapshot/ipc/file/enabled"_json_pointer, false)) {
    ipc->enableFile(
        evaluateStaticVariables(conf["snapshot"]["ipc"]["file"]["path"]));
  }
  if (conf.value("/snapshot/ipc/zeroMQ/enabled"_json_pointer, false)) {
    ipc->enableZeroMQ(
        evaluateStaticVariables(conf["snapshot"]["ipc"]["zeroMQ"]["endpoint"]));
  }
  if (conf.value("/snapshot/ipc/sharedMem/enabled"_json_pointer, false)) {
    ipc->enableSharedMemory(
        evaluateStaticVariables(
            conf["snapshot"]["ipc"]["sharedMem"]["sharedMemName"]),
        conf["snapshot"]["ipc"]["sharedMem"]["sharedMemSize"].get<size_t>(),
        evaluateStaticVariables(
            conf["snapshot"]["ipc"]["sharedMem"]["semaphoreName"]));
  }
}

string DeviceManager::evaluateVideoSpecficVariables(
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
DeviceManager::evaluateStaticVariables(basic_string<char> originalString) {
  string filledString = regex_replace(
      originalString, regex(R"(\{\{deviceIndex\}\})"), to_string(deviceIndex));
  filledString = regex_replace(filledString, regex(R"(\{\{deviceName\}\})"),
                               conf["name"].get<string>());
  return filledString;
}

void DeviceManager::setParameters(const size_t deviceIndex) {

  // Most config items will be directly used from njson object, however,
  // given performance concern, for items that are on the critical path,
  // we will duplicate them as class member variables
  this->deviceIndex = deviceIndex;
  conf["name"] = evaluateStaticVariables(conf["name"]);
  deviceName = conf["name"];
  conf["videoFeed"]["uri"] = evaluateStaticVariables(conf["videoFeed"]["uri"]);

  conf["events"]["onVideoStarts"] =
      evaluateStaticVariables(conf["events"]["onVideoStarts"]);
  conf["events"]["onVideoEnds"] =
      evaluateStaticVariables(conf["events"]["onVideoEnds"]);
  conf["events"]["onDeviceOffline"] =
      evaluateStaticVariables(conf["events"]["onDeviceOffline"]);
  conf["events"]["onDeviceBackOnline"] =
      evaluateStaticVariables(conf["events"]["onDeviceBackOnline"]);

  // ===== frame =====
  frameRotation = conf["frame"]["rotation"];
  outputWidth = conf["frame"]["outputWidth"];
  outputHeight = conf["frame"]["outputHeight"];
  throttleFpsIfHigherThan = conf["frame"]["throttleFpsIfHigherThan"];
  textOverlayEnabled = conf["frame"]["textOverlay"]["enabled"];
  textOverlayFontSacle = conf["frame"]["textOverlay"]["fontScale"];
  frameQueueSize = conf["frame"]["queueSize"];

  // =====  snapshot =====
  snapshotFrameInterval = conf["snapshot"]["frameInterval"];

  // ===== motion detection =====
  motionDetectionMode = conf["motionDetection"]["mode"];
  frameDiffPercentageLowerLimit =
      conf["motionDetection"]["frameDiffPercentageLowerLimit"];
  frameDiffPercentageUpperLimit =
      conf["motionDetection"]["frameDiffPercentageUpperLimit"];
  pixelDiffAbsThreshold = conf["motionDetection"]["pixelDiffAbsThreshold"];
  diffEveryNthFrame = conf["motionDetection"]["diffEveryNthFrame"];
  if (diffEveryNthFrame == 0) {
    throw invalid_argument("diffEveryNthFrame must be greater than 0");
  }
  minFramesPerVideo =
      conf["motionDetection"]["videoRecording"]["minFramesPerVideo"];
  maxFramesPerVideo =
      conf["motionDetection"]["videoRecording"]["maxFramesPerVideo"];

  conf["motionDetection"]["videoRecording"]["videoWriter"]["videoPath"] =
      evaluateStaticVariables(conf["motionDetection"]["videoRecording"]
                                  ["videoWriter"]["videoPath"]);
  drawContours = conf["motionDetection"]["drawContours"];

  spdlog::info("{}-th device to be used with the following configs:\n{}",
               deviceIndex, conf.dump(2));
}

DeviceManager::~DeviceManager() {}

float DeviceManager::getCurrentFps(int64_t msSinceEpoch) {
  float fps = FLT_MAX;
  if (msSinceEpoch - frameTimestamps.front() > 0) {
    fps = 1000.0 * frameTimestamps.size() /
          (msSinceEpoch - frameTimestamps.front());
  }
  return fps;
}

bool DeviceManager::shouldFrameBeThrottled() {

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

void DeviceManager::stopVideoRecording(VideoWriter &vwriter,
                                       uint32_t &videoFrameCount, int cd) {

  vwriter.release();

  if (cd > 0) {
    spdlog::warn("[{}] video recording stopped before cooldown reaches 0",
                 deviceName);
  }

  if (conf["events"]["onVideoEnds"].get<string>().size() > 0 && cd == 0) {
    spdlog::info("[{}] onVideoEnds triggered", deviceName);
    execExternalProgramAsync(mtxOnVideoEnds,
                             evaluateVideoSpecficVariables(
                                 conf["events"]["onVideoEnds"].get<string>()),
                             deviceName);
  } else if (conf["events"]["onVideoEnds"].get<string>().size() > 0 && cd > 0) {
    spdlog::warn("[{}] onVideoEnds event defined but it won't be triggered",
                 deviceName);
  } else {
    spdlog::info("[{}] onVideoEnds triggered but no command to execute",
                 deviceName);
  }
  videoFrameCount = 0;
}

void DeviceManager::startOrKeepVideoRecording(VideoWriter &vwriter,
                                              int64_t &cd) {

  auto handleOnVideoStarts = [&]() {
    if (conf["events"]["onVideoStarts"].get<string>().size() > 0) {
      spdlog::info("[{}] motion detected, video recording begins", deviceName);
      execExternalProgramAsync(
          mtxOnVideoStarts,
          evaluateVideoSpecficVariables(
              conf["events"]["onVideoStarts"].get<string>()),
          deviceName);
      // execAsync((void *)this, args, asyncExecCallback);
    } else {
      spdlog::info("[{}] onVideoStarts triggered but no command to execute",
                   deviceName);
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

void DeviceManager::getLiveImage(vector<uint8_t> &pl) {
  if (ipc->encodedJpgImage.size() > 0) {
    lock_guard<mutex> guard(mutexLiveImage);
    pl = ipc->encodedJpgImage;
  } else {
    pl = vector<uint8_t>();
  }
}

void DeviceManager::updateVideoCooldownAndVideoFrameCount(
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

void DeviceManager::markDeviceAsOffline(bool &isShowingBlankFrame) {

  if (isShowingBlankFrame == false) {
    timestampOnDeviceOffline = getCurrentTimestamp();
    isShowingBlankFrame = true;
    if (conf["events"]["onDeviceOffline"].get<string>().size() > 0) {
      spdlog::info("[{}] onDeviceOffline triggered", deviceName);
      execExternalProgramAsync(
          mtxOnDeviceOffline,
          evaluateVideoSpecficVariables(
              conf["events"]["onDeviceOffline"].get<string>()),
          deviceName);
    } else {
      spdlog::info("[{}] onDeviceOffline triggered but no command to execute",
                   deviceName);
    }
  }
}

void DeviceManager::deviceIsBackOnline(size_t &openRetryDelay,
                                       bool &isShowingBlankFrame) {
  spdlog::info("[{}] Device is back online", deviceName);
  timestampOnDeviceOffline = "";
  openRetryDelay = 1;
  isShowingBlankFrame = false;
  if (conf["events"]["onDeviceBackOnline"].get<string>().size() > 0) {
    spdlog::info("[{}] onDeviceBackOnline triggered", deviceName);
    execExternalProgramAsync(
        mtxOnDeviceBackOnline,
        evaluateVideoSpecficVariables(
            conf["events"]["onDeviceBackOnline"].get<string>()),
        deviceName);
  } else {
    spdlog::info("[{}] onDeviceBackOnline triggered but no command to execute",
                 deviceName);
  }
}

void DeviceManager::initializeDevice(VideoCapture &cap, bool &result,
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

void DeviceManager::warnCPUResize(const Size &actualFrameSize) {
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

void DeviceManager::InternalThreadEntry() {

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
      FH::generateBlankFrameAt1Fps(currFrame, actualFrameSize);
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
      ipc->sendData(dispFrames.front());
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
