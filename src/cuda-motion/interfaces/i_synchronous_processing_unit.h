#pragma once

#include "../entities/synchronous_processing_result.h"
#include "../entities/processing_metadata.h"

#include <nlohmann/json.hpp>
#include <opencv2/cudawarping.hpp>

using njson = nlohmann::json;

namespace CudaMotion::ProcessingUnit {

class ISynchronousProcessingUnit {
public:
  virtual ~ISynchronousProcessingUnit() = default;

  virtual bool init(const njson &config) = 0;

  ///
  /// @param frame the frame to be processed
  /// @return
  virtual SynchronousProcessingResult process(cv::cuda::GpuMat &frame, PipelineContext& meta_data) = 0;
};
}