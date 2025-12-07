#pragma once

#include "entities/processing_metadata.h"
#include "event_loop.h"

#include <nlohmann/json.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudacodec.hpp>

#include <queue>
#include <string>

using njson = nlohmann::json;

namespace CudaMotion {
struct videoWritingContext {
  std::string evaluatedVideoPath;
  float fps;
};

class DeviceManager : public EventLoop {

public:
  DeviceManager();
  ~DeviceManager();
  std::string getDeviceName() { return this->deviceName; }

protected:
  void InternalThreadEntry() override;

private:
  std::string deviceName;

  // frame variables
  std::string evaluatedVideoPath;

  std::string timestampOnVideoStarts;
  std::string timestampOnDeviceOffline;
  // moodycamel::ReaderWriterQueue<uint64_t> frameTimestamps;
  std::deque<uint64_t> frameTimestamps;
  std::mutex mtx_vr;
  std::atomic<bool> delayed_vc_open_retry_registered{false};

  void always_fill_in_frame(const cv::Ptr<cv::cudacodec::VideoReader> &vr,
                            int expected_frame_height, int expected_frame_width,
                            cv::cuda::GpuMat &frame,
                            ProcessingUnit::PipelineContext &ctx);
  void handle_video_capture(cv::Ptr<cv::cudacodec::VideoReader> &vr,
                            const ProcessingUnit::PipelineContext &ctx,
                            const std::string &video_feed);
};
} // namespace CudaMotion
