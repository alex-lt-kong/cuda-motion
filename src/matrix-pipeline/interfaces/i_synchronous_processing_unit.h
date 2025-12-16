#pragma once

#include "../entities/processing_context.h"
#include "../entities/synchronous_processing_result.h"

#include <nlohmann/json.hpp>
#include <opencv2/cudawarping.hpp>

using njson = nlohmann::json;

namespace MatrixPipeline::ProcessingUnit {

class ISynchronousProcessingUnit {
public:
  virtual ~ISynchronousProcessingUnit() = default;

  virtual bool init(const njson &config) = 0;

  ///
  /// @param frame the frame to be processed
  /// @return
  virtual SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                              PipelineContext &ctx) = 0;
};


class DebugOutput final : public ISynchronousProcessingUnit {
  std::string m_custom_text;
public:
  DebugOutput() = default;
  ~DebugOutput() override = default;

  bool init(const njson &config) override;

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx) override;
};
} // namespace MatrixPipeline::ProcessingUnit