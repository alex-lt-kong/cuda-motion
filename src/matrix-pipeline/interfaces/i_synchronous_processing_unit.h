#pragma once

#include "../entities/processing_context.h"
#include "../entities/synchronous_processing_result.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

using njson = nlohmann::json;

namespace MatrixPipeline::ProcessingUnit {

class ISynchronousProcessingUnit {
protected:
  const std::string m_unit_path;

public:
  explicit ISynchronousProcessingUnit(std::string unit_path)
      : m_unit_path(std::move(unit_path)) {
    SPDLOG_INFO("Initializing synchronous_processing_unit: {}", m_unit_path);
  };
  virtual ~ISynchronousProcessingUnit() {
    SPDLOG_INFO("synchronous_processing_unit {} destructed", m_unit_path);
  }

  virtual bool init(const njson &config) = 0;

  ///
  /// @param frame the frame to be processed
  /// @param ctx context
  /// @return
  virtual SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                              PipelineContext &ctx) = 0;
};

} // namespace MatrixPipeline::ProcessingUnit