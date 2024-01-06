#include "device_manager.h"
#include "frame_handler.h"
#include "global_vars.h"

#include <cstddef>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/videoio.hpp>
#include <spdlog/spdlog.h>

#include <errno.h>
#include <filesystem>
#include <limits.h>
#include <regex>
#include <sstream>
#include <sys/socket.h>

using namespace std;

DeviceManager::DeviceManager(const size_t deviceIndex) : pt(10000) {

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
        evaluateStaticVariables(
            conf.value("/snapshot/ipc/zeroMQ/endpoint"_json_pointer,
                       "tcp://127.0.0.1:4240")),
        conf.value("/snapshot/ipc/zeroMQ/sendCVMat"_json_pointer, false));
  }
  if (conf.value("/snapshot/ipc/sharedMem/enabled"_json_pointer, false)) {
    ipc->enableSharedMemory(
        evaluateStaticVariables(
            conf["snapshot"]["ipc"]["sharedMem"]["sharedMemName"]),
        conf["snapshot"]["ipc"]["sharedMem"]["sharedMemSize"].get<size_t>(),
        evaluateStaticVariables(
            conf["snapshot"]["ipc"]["sharedMem"]["semaphoreName"]));
  }
  fh = make_unique<FrameHandler::FrameHandler>(
      conf["frame"]["textOverlay"]["fontScale"].get<double>(), deviceName);
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

  // ===== frame =====

  frameRotationAngle = conf.value("/frame/rotationAngle"_json_pointer, 0.0);
  textOverlayEnabled =
      conf.value("/frame/textOverlay/enabled"_json_pointer, false);
  frameQueueSize = conf.value("/frame/queueSize"_json_pointer, 5);
  assert(frameQueueSize > 0);

  // =====  snapshot =====
  snapshotFrameInterval = conf["snapshot"]["frameInterval"];

  // ===== events =====
  conf["events"]["onVideoStarts"] =
      evaluateStaticVariables(conf["events"]["onVideoStarts"]);
  conf["events"]["onVideoEnds"] =
      evaluateStaticVariables(conf["events"]["onVideoEnds"]);
  conf["events"]["onDeviceOffline"] =
      evaluateStaticVariables(conf["events"]["onDeviceOffline"]);
  conf["events"]["onDeviceBackOnline"] =
      evaluateStaticVariables(conf["events"]["onDeviceBackOnline"]);

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

DeviceManager::~DeviceManager() {
  ipc->wait();
  vwPcQueue.wait();
}

float DeviceManager::getCurrentFps() {

  // std::queue seems to be causing performance issue, let give
  // moodycamel::ReaderWriterQueue<uint64_t> a try!
  // int64 start = chrono::duration_cast<chrono::microseconds>(
  //                  chrono::system_clock::now().time_since_epoch())
  //                  .count();
  constexpr int sampleMsUpperLimit = 30 * 1000;
  auto msSinceEpoch = chrono::duration_cast<chrono::milliseconds>(
                          chrono::system_clock::now().time_since_epoch())
                          .count();
  frameTimestamps.push_back(msSinceEpoch);
  uint64_t front = frameTimestamps.front();
  if (msSinceEpoch - front > sampleMsUpperLimit) {
    frameTimestamps.pop_front();
  }

  float fps = 0;
  if (msSinceEpoch - front > 0) {
    fps = 1000.0 * frameTimestamps.size() / (msSinceEpoch - front);
  }
  /*int64 end = chrono::duration_cast<chrono::microseconds>(
                  chrono::system_clock::now().time_since_epoch())
                  .count();
  pt.addSample(end - start);
  if (pt.totalSampleCount() % 1000 == 0) {
    pt.refreshStats();
    double percentiles[] = {50, 66, 90, 95, 100};
    spdlog::info("[{}] Percentiles in us (sampleCount: {})", deviceName,
                 pt.sampleCount());
    for (size_t i = 0; i < sizeof(percentiles) / sizeof(percentiles[0]); ++i) {
      spdlog::info("{:3}-th: {:6}", percentiles[i],
                   pt.getPercentile(percentiles[i]));
    }
    spdlog::info("Average: {:5}", (long)pt.getAverage());
  }
  */
  // Profiling result is not conslusive: std::queue<T> is nothing but a wrapper
  // over std::deque<T> so it is impossible for std::queue<T> to be faster than
  // std::deque<T>

  // moodycamel::ReaderWriterQueue<uint64_t> frameTimestamps;
  //  Percentiles in us (sampleCount: 10000)
  //   50-th:     49
  //   66-th:     57
  //   90-th:     78
  //   95-th:    122
  //  100-th:  12730

  // std::queue<uint64_t>
  // Percentiles in us (sampleCount: 10000)
  //  50-th:     55
  //  66-th:     59
  //  90-th:     73
  //  95-th:     86
  //  100-th:   4845
  //  Average:    58

  // std::deque<uint64_t>
  //  Percentiles in us (sampleCount: 10000)
  //  50-th:     55
  //  66-th:     61
  //  90-th:     89
  //  95-th:    169
  //  100-th:  12707
  //  Average:    86
  return fps;
}

void DeviceManager::stopVideoRecording(uint32_t &videoFrameCount, int cd) {
  if (videoWriting) {
    videoWriting = false;
    vwPcQueue.wait();
    spdlog::info(
        "[{}]  vwPcQueue thread exited, video recording stopped gracefully",
        deviceName);
  }

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

void DeviceManager::startOrKeepVideoRecording(int64_t &cd) {

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
  if (videoWriting)
    return;

  timestampOnVideoStarts = getCurrentTimestamp();
  evaluatedVideoPath = evaluateVideoSpecficVariables(
      conf["motionDetection"]["videoRecording"]["videoWriter"]["videoPath"]);

  handleOnVideoStarts();
  videoWriting = true;

  vwPcQueue.start(
      {.evaluatedVideoPath = evaluatedVideoPath,
       .fps = conf["motionDetection"]["videoRecording"]["videoWriter"]["fps"],
       .outputWidth = outputWidth,
       .outputHeight = outputHeight,
       .videoWriting = videoWriting});
}

void DeviceManager::getLiveImage(vector<uint8_t> &pl) {
  if (ipc->isHttpEnabled() && ipc->encodedJpgImage.size() > 0) {
    lock_guard<mutex> guard(mutexLiveImage);
    pl = ipc->encodedJpgImage;
  } else {
    pl = vector<uint8_t>();
  }
}

void DeviceManager::updateVideoCooldownAndVideoFrameCount(
    int64_t &cd, uint32_t &videoFrameCount) noexcept {
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

void DeviceManager::InternalThreadEntry() {

  queue<Mat> hDispFrames;
  cuda::GpuMat dPrevFrame, dCurrFrame, dDiffFrame;
  bool result = false;
  bool isShowingBlankFrame = false;
  // VideoCapture cap;
  Ptr<cudacodec::VideoReader> vr;
  float rateOfChange = 0.0;

  uint64_t frameCountSinceLastOpen = 0;
  uint64_t frameCountSinceStart = 0;
  uint32_t videoFrameCount = 0;
  // Will be updated after 1st actual frame is received
  Size actualFrameSize = Size(1280, 720);

  // cd: cooldown
  int64_t cd = 0;
  size_t openRetryDelay = 1;

  // We use the evil `goto` statement so that we can avoid the duplication of
  //   initializeDevice()...
  goto entryPoint;

  while (ev_flag == 0) {
    if (vr != nullptr) [[likely]] {
      result = vr->nextFrame(dCurrFrame);
    } else {
      result = false;
      // assume the video source is at 30fps
      this_thread::sleep_for(chrono::milliseconds(1000 / 30));
    }
    ++frameCountSinceStart;
    ++frameCountSinceLastOpen;

    if (result == false || dCurrFrame.empty()) [[unlikely]] {
      markDeviceAsOffline(isShowingBlankFrame);
      fh->generateBlankFrameAt1Fps(dCurrFrame, actualFrameSize);
      if (frameCountSinceStart % openRetryDelay == 0) {
        openRetryDelay *= 2;
        spdlog::error("[{}] VideoReader failed to read a new frame, "
                      "waiting for {} reading attempts then retry...",
                      deviceName, openRetryDelay);
      entryPoint:
        try {
          vr = cudacodec::createVideoReader(
              conf["videoFeed"]["uri"].get<string>());
          vr->set(cudacodec::ColorFormat::BGR);
          spdlog::info("[{}] VideoReader initialized ({})", deviceName,
                       conf["videoFeed"]["uri"].get<string>());
        } catch (const cv::Exception &e) {
          spdlog::error("[{}] VideoReader initialization failed: {}",
                        deviceName, e.what());
        }
        frameCountSinceLastOpen = 0;
        // If they are -1, we want to reload them
        outputWidth = conf.value("/frame/outputWidth"_json_pointer, 1280);
        outputHeight = conf.value("/frame/outputHeight"_json_pointer, 720);
        continue;
      }
    } else if (isShowingBlankFrame) [[unlikely]] {
      deviceIsBackOnline(openRetryDelay, isShowingBlankFrame);
    }
    if (frameRotationAngle != 0.0 && isShowingBlankFrame == false) {
      cv::cuda::GpuMat gpuRotatedImage;
      cv::cuda::rotate(dCurrFrame.clone(), dCurrFrame, dCurrFrame.size(),
                       frameRotationAngle, static_cast<float>(dCurrFrame.cols),
                       static_cast<float>(dCurrFrame.rows));
    }
    if (frameCountSinceLastOpen == 1) {
      actualFrameSize = dCurrFrame.size();
      outputWidth = outputWidth == -1 ? actualFrameSize.width : outputWidth;
      outputHeight = outputHeight == -1 ? actualFrameSize.height : outputHeight;
    }

    if (motionDetectionMode == MODE_DETECT_MOTION &&
        frameCountSinceStart % diffEveryNthFrame == 0) {
      if (isShowingBlankFrame == false) {
        // profiling shows this if() block takes around 1-2 ms
        rateOfChange = fh->getFrameChanges(dPrevFrame, dCurrFrame, dDiffFrame,
                                           pixelDiffAbsThreshold);
      } else {
        rateOfChange = -1;
      }
      /* Can't just assign like prevFrame = currFrame, otherwise two
      objects will share the same copy of underlying image data */
      // dPrevFrame = dCurrFrame;
      dCurrFrame.copyTo(dPrevFrame);
    }
    if (actualFrameSize.width != outputWidth ||
        actualFrameSize.height != outputHeight) {
      cuda::resize(dCurrFrame, dCurrFrame, cv::Size(outputWidth, outputHeight));
    }
    // cv::Mat behaves like a std::shared_ptr<T>
    Mat hFrame;
    dCurrFrame.download(hFrame);
    hDispFrames.push(hFrame);
    if (hDispFrames.size() > frameQueueSize) {
      hDispFrames.pop();
    }
    if (drawContours && motionDetectionMode == MODE_DETECT_MOTION &&
        isShowingBlankFrame == false) {
      Mat hDiffFrame;
      dDiffFrame.download(hDiffFrame);
      fh->overlayContours(hFrame, hDiffFrame);
      // CPU-intensive! Use with care!
    }
    if (textOverlayEnabled) {
      fh->overlayStats(hFrame, rateOfChange, cd, videoFrameCount,
                       getCurrentFps(), maxFramesPerVideo);
      fh->overlayDatetime(hFrame, timestampOnDeviceOffline);
      fh->overlayDeviceName(hFrame);
    }

    if ((frameCountSinceStart - 1) % snapshotFrameInterval == 0) {
      ipc->enqueueData(hDispFrames.front().clone());
    }

    if (motionDetectionMode == MODE_DISABLED) {
      continue;
    }

    if (motionDetectionMode == MODE_ALWAYS_RECORD ||
        (rateOfChange > frameDiffPercentageLowerLimit &&
         rateOfChange < frameDiffPercentageUpperLimit &&
         motionDetectionMode == MODE_DETECT_MOTION)) {
      startOrKeepVideoRecording(cd);
    }

    updateVideoCooldownAndVideoFrameCount(cd, videoFrameCount);

    if (cd < 0) {
      continue;
    }
    if (cd == 0) {
      stopVideoRecording(videoFrameCount, cd);
      continue;
    }
    cuda::GpuMat dDispFrame;
    dDispFrame.upload(hDispFrames.front());
    if (!vwPcQueue.try_enqueue(dDispFrame)) {
      spdlog::warn("[{}] pcQueue is full", deviceName);
    }
    // enqueueVideoWriterFrame(dispFrames.front());
  }
  if (cd > 0) {
    stopVideoRecording(videoFrameCount, cd);
  }
  vr.release();
  spdlog::info("[{}] thread quits gracefully", deviceName);
}
