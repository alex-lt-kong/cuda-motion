#pragma once

#include "../entities/processing_context.h"
#include "../entities/processing_units_variant.h"

#include <nlohmann/json.hpp>

using njson = nlohmann::json;

namespace MatrixPipeline::ProcessingUnit {

class PipelineExecutor final: public ISynchronousProcessingUnit{

  std::vector<ProcessingUnitVariant> m_processing_units;

public:
  PipelineExecutor() = default;

  ~PipelineExecutor() override = default;

  bool init(const njson &settings) override;

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                      PipelineContext &ctx) override;
};
}