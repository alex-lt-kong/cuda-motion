#pragma once

#include "../entities/processing_context.h"
#include "../entities/synchronous_processing_result.h"
#include "i_processing_unit.h"

namespace MatrixPipeline::ProcessingUnit {

class ISynchronousProcessingUnit : public IProcessingUnit {

public:
  explicit ISynchronousProcessingUnit(const std::string &unit_path)
      : IProcessingUnit(unit_path) {};

  ///
  /// @param frame the frame to be processed
  /// @param ctx context
  /// @return
  virtual SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                              PipelineContext &ctx) = 0;
};

} // namespace MatrixPipeline::ProcessingUnit