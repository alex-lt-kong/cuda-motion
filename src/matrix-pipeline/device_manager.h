#pragma once

#include "entities/processing_context.h"
#include "event_loop.h"

#include <nlohmann/json.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudacodec.hpp>

#include <string>

using njson = nlohmann::json;

namespace MatrixPipeline {

struct videoWritingContext {
  std::string evaluatedVideoPath;
  float fps;
};

class DeviceManager : public EventLoop {

public:
  DeviceManager() = default;
  ~DeviceManager() override {}
  std::string getDeviceName() { return this->deviceName; }

protected:
  void InternalThreadEntry() override;

private:
  std::string deviceName;

  std::mutex mtx_vr;
  std::atomic<bool> delayed_vc_open_retry_registered{false};
  cv::Ptr<cv::cudacodec::VideoReader> vr{nullptr};
  void always_fill_in_frame(cv::cuda::GpuMat &frame,
                            ProcessingUnit::PipelineContext &ctx);
  void handle_video_capture(const ProcessingUnit::PipelineContext &ctx);
};
} // namespace MatrixPipeline
