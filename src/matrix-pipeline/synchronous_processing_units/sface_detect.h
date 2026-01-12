#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

#include <opencv2/objdetect.hpp>
#include <string>
#include <vector>

namespace MatrixPipeline::ProcessingUnit {

class SfaceDetect : public ISynchronousProcessingUnit {
public:
  explicit SfaceDetect(const std::string &unit_path)
      : ISynchronousProcessingUnit(unit_path + "/SfaceDetect") {}

  ~SfaceDetect() override = default;

  bool init(const nlohmann::json &config) override;

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx) override;

private:
  struct Identity {
    std::string name;
    std::vector<cv::Mat> embeddings;
  };

  cv::Ptr<cv::FaceRecognizerSF> m_sface;

  // Configs
  std::string m_model_path_sface;
  std::string m_model_path_yunet;
  std::string m_gallery_directory;
  float m_match_threshold{0.363};
  std::chrono::milliseconds m_inference_interval{100};
  std::chrono::time_point<std::chrono::steady_clock> m_last_inference_at;
  SFaceContext m_prev_sface_ctx;
  std::vector<Identity> m_gallery;

  // Helper: Returns false if gallery cannot be populated
  bool load_gallery();
};

} // namespace MatrixPipeline::ProcessingUnit