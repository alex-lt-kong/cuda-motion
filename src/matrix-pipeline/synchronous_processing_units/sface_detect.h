#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

#include <opencv2/objdetect.hpp>
#include <string>
#include <vector>

namespace MatrixPipeline::ProcessingUnit {

class SfaceDetect : public ISynchronousProcessingUnit {
public:
  explicit SfaceDetect(const std::string &unit_path);
  ~SfaceDetect() override = default;

  bool init(const nlohmann::json &config) override;

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx) override;

private:
  cv::Ptr<cv::FaceRecognizerSF> m_sface;

  // Configs
  std::string m_model_path_sface;
  std::string m_model_path_yunet;
  std::string m_gallery_directory;
  float m_match_threshold;
  // In-memory gallery: pair<Name, Embedding>
  std::vector<std::pair<std::string, cv::Mat>> m_gallery;

  // Helper: Returns false if gallery cannot be populated
  bool load_gallery();
};

} // namespace MatrixPipeline::ProcessingUnit