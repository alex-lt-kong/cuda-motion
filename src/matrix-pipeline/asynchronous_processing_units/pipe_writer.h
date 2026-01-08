#pragma once

#include "../interfaces/i_asynchronous_processing_unit.h"

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include <cstdio>
#include <string>

namespace MatrixPipeline::ProcessingUnit {

using njson = nlohmann::json;

class PipeWriter : public IAsynchronousProcessingUnit {
public:
  explicit PipeWriter(const std::string &unit_path)
      : IAsynchronousProcessingUnit(unit_path + "/FFmpegStreamer") {}
  ~PipeWriter() override;

  bool init(const njson &config) override;

protected:
  void on_frame_ready(cv::cuda::GpuMat &gpu_frame,
                      PipelineContext &ctx) override;

private:
  void close_pipe() noexcept;

  std::string m_subprocess_cmd;
  FILE *m_pipe = nullptr;
  cv::Mat m_cpu_frame; // Reusable CPU buffer to avoid re-allocation
};

} // namespace MatrixPipeline::ProcessingUnit