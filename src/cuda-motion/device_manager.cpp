#include "device_manager.h"
#include "frame_handler.h"
#include "global_vars.h"
#include "interfaces/i_synchronous_processing_unit.h"
#include "ipc.h"
#include "synchronous_processing_units/calculate_change_rate.h"
#include "synchronous_processing_units/calculate_fps.h"
#include "synchronous_processing_units/crop_frame.h"
#include "synchronous_processing_units/overlay_info.h"
#include "synchronous_processing_units/rotate_frame.h"

#include <cstddef>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/cudacodec.hpp>
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
using namespace CudaMotion;

DeviceManager::DeviceManager(const size_t deviceIndex)
    : pt(10000), vwPcQueue(&ev_flag, 512) {

  { // settings variable is used by multiple threads, possibly concurrently, is
    // reading it thread-safe? According to the below response from the library
    // author: https://github.com/nlohmann/json/issues/651
    // The answer seems to be affirmative.
    // But writing to settings is unlikely to be thread-safe:
    // https://github.com/nlohmann/json/issues/651
    lock_guard<mutex> guard(mtxNjsonSettings);
    const auto t = settings["devices"][deviceIndex];
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
  fh = make_unique<Utils::FrameHandler>(
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
                               Utils::getCurrentTimestamp());
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

  frameRotationAngle = conf.value("/frame/rotationAngle"_json_pointer, 0);
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

  spdlog::info("{}-th device to be used with the following configs:\n{}",
               deviceIndex, conf.dump(2));
}

DeviceManager::~DeviceManager() {
  ipc->wait();
  vwPcQueue.wait();
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
    Utils::execExternalProgramAsync(
        mtxOnVideoEnds,
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
    if (!conf["events"]["onVideoStarts"].get<string>().empty()) {
      spdlog::info("[{}] motion detected, video recording begins", deviceName);
      Utils::execExternalProgramAsync(
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

  timestampOnVideoStarts = Utils::getCurrentTimestamp();
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

string DeviceManager::getLiveImageBytes() {
  if (ipc->isHttpEnabled()) {
    lock_guard<mutex> guard(mutexLiveImage);
    return ipc->getJpegBytes();
  } else {
    return string();
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
    timestampOnDeviceOffline = Utils::getCurrentTimestamp();
    isShowingBlankFrame = true;
    if (conf["events"]["onDeviceOffline"].get<string>().size() > 0) {
      spdlog::info("[{}] onDeviceOffline triggered", deviceName);
      Utils::execExternalProgramAsync(
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

void DeviceManager::deviceIsBackOnline(size_t &retryInterval,
                                       bool &isShowingBlankFrame) {
  spdlog::info("[{}] Device is back online", deviceName);
  timestampOnDeviceOffline = "";
  retryInterval = snapshotFrameInterval + 1;
  isShowingBlankFrame = false;
  if (conf["events"]["onDeviceBackOnline"].get<string>().size() > 0) {
    spdlog::info("[{}] onDeviceBackOnline triggered", deviceName);
    Utils::execExternalProgramAsync(
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
  using namespace ProcessingUnit;
  std::vector<std::unique_ptr<ISynchronousProcessingUnit>>
      sync_processing_units;
  for (nlohmann::basic_json<>::size_type i = 0; i < settings["pipeline"].size();
       ++i) {
    settings["pipeline"][i]["device"] = settings["device"];
    std::unique_ptr<ISynchronousProcessingUnit> ptr;
    if (settings["pipeline"][i]["type"].get<std::string>() == "rotation") {
      ptr = std::make_unique<RotateFrame>();
    } else if (settings["pipeline"][i]["type"].get<std::string>() ==
               "overlayInfo") {
      ptr = std::make_unique<OverlayInfo>();
    } else if (settings["pipeline"][i]["type"].get<std::string>() ==
               "cropFrame") {
      ptr = std::make_unique<CropFrame>();
    } else if (settings["pipeline"][i]["type"].get<std::string>() ==
               "calculateChangeRate") {
      ptr = std::make_unique<CalculateChangeRate>();
    } else if (settings["pipeline"][i]["type"].get<std::string>() ==
               "calculateFps") {
      ptr = std::make_unique<CalculateFps>();
    } else {
      continue;
    }
    sync_processing_units.push_back(std::move(ptr));
    sync_processing_units.back()->init(settings["pipeline"][i]);
  }

  queue<Mat> hDispFrames;
  cuda::GpuMat dPrevFrame, dCurrFrame, dDiffFrame;
  bool result = false;
  bool dummyResult = false;
  bool isShowingBlankFrame = false;
  // VideoCapture cap;
  Ptr<cudacodec::VideoReader> vr;
  const auto video_feed = settings["device"]["uri"].get<std::string>();
  // conf.value("/videoFeed/uri"_json_pointer, "/dev/video0");

  uint64_t frameCountSinceLastOpen = 0;
  uint64_t frameCountSinceStart = 0;
  uint32_t videoFrameCount = 0;
  // Will be updated after 1st actual frame is received
  Size actualFrameSize = Size(1280, 720);

  // cd: cooldown
  int64_t cd = 0;
  size_t retryInterval = snapshotFrameInterval + 1;
  ProcessingMetaData meta_data;
  // We use the evil `goto` statement so that we can avoid the duplication of
  //   initializeDevice()...
  goto entryPoint;

  while (ev_flag == 0) {
    if (vr != nullptr) [[likely]] {
      try {
        result = vr->nextFrame(dCurrFrame);
      } catch (const cv::Exception &e) {
        result = false;
        spdlog::error("[{}] VideoReader nextFrame() failed: {}", deviceName,
                      e.what());
      }
    } else {
      result = false;
    }
    ++frameCountSinceStart;
    ++frameCountSinceLastOpen;

    if (result == false || dCurrFrame.empty()) [[unlikely]] {
      markDeviceAsOffline(isShowingBlankFrame);
      dummyResult = fh->nextDummyFrame(dCurrFrame, actualFrameSize);
      if (frameCountSinceStart % retryInterval == 0 || !dummyResult) {
        if (retryInterval < 360)
          retryInterval *= 2;
        spdlog::warn("[{}] Retry initializing VideoReader with videoFeed [{}] "
                     "after {} consecutive reading attempts ...",
                     deviceName, video_feed, retryInterval);
      entryPoint:
        auto params = cudacodec::VideoReaderInitParams();
        // https://docs.opencv.org/4.9.0/dd/d7d/structcv_1_1cudacodec_1_1VideoReaderInitParams.html#a9b73352d9bc1a23ccf3bf06517f978c7
        params.allowFrameDrop = true;
        try {
          // Explicitly release the OLD resource first
          if (vr) {
            vr.release(); // Decrements refcount, forces destructor NOW
            vr = nullptr; // Safety
          }
          vr = cudacodec::createVideoReader(video_feed, {}, params);
          vr->set(cudacodec::ColorFormat::BGR);
          spdlog::info("[{}] VideoReader initialized ({})", deviceName,
                       video_feed);
        } catch (const cv::Exception &e) {
          spdlog::error("[{}] cudacodec::createVideoReader({}) failed: {}",
                        deviceName, video_feed, e.what());
          // vr's RAII is not synchronous, it delegates the real work to GPU
          std::this_thread::sleep_for(std::chrono::seconds(2));
        }

        frameCountSinceLastOpen = 0;
        // If they are -1, we want to reload them
        outputWidth = conf.value("/frame/outputWidth"_json_pointer, 1280);
        outputHeight = conf.value("/frame/outputHeight"_json_pointer, 720);
        continue;
      }
    } else if (isShowingBlankFrame) [[unlikely]] {
      deviceIsBackOnline(retryInterval, isShowingBlankFrame);
    }

    meta_data.capture_timestamp_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    for (size_t i = 0; i < sync_processing_units.size(); ++i) {
      sync_processing_units[i]->process(dCurrFrame, meta_data);
    }
    if (frameCountSinceLastOpen == 1) {
      actualFrameSize = dCurrFrame.size();
      outputWidth = outputWidth == -1 ? actualFrameSize.width : outputWidth;
      outputHeight = outputHeight == -1 ? actualFrameSize.height : outputHeight;
    }

    if (motionDetectionMode == MODE_DETECT_MOTION &&
        frameCountSinceStart % diffEveryNthFrame == 0) {
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

    if (frameCountSinceStart % snapshotFrameInterval == 0 ||
        frameCountSinceStart < 4 /* Use to handle initial failure*/) {
      ipcQueueElement pl = {.rateOfChange = meta_data.change_rate,
                            .cooldown = cd,
                            .snapshot = hDispFrames.front().clone()};
      ipc->enqueueData(pl);
    }

    if (motionDetectionMode == MODE_DISABLED) {
      continue;
    }

    if (motionDetectionMode == MODE_ALWAYS_RECORD ||
        (meta_data.change_rate > frameDiffPercentageLowerLimit &&
         meta_data.change_rate < frameDiffPercentageUpperLimit &&
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

    try {
      cuda::GpuMat dDispFrame;
      // there involves a GPU memory allocation
      dDispFrame.upload(hDispFrames.front());
      if (!vwPcQueue.try_enqueue(dDispFrame)) {
        spdlog::warn("[{}] pcQueue is full", deviceName);
      }
    } catch (const Exception &e) {
      spdlog::error(
          "[{}] dDispFrame.upload() failed, the event loop will exit, what: {}",
          deviceName, e.what());
      ev_flag = 1;
      break;
    }
  }
  if (cd > 0) {
    stopVideoRecording(videoFrameCount, cd);
  }
  vr.release();
  spdlog::info("[{}] thread quits gracefully", deviceName);
}
