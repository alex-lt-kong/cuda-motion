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
    std::vector<cv::Mat> normalized_embeddings;
    IdentityCategory category;
  };

  cv::Ptr<cv::FaceRecognizerSF> m_sface;

  // Configs
  double m_authorized_enrollment_face_score_threshold{0.93};
  double m_unauthorized_enrollment_face_score_threshold{0.60};
  std::optional<double> m_inference_face_score_threshold{std::nullopt};
  std::string m_model_path_sface;
  std::string m_model_path_yunet;
  std::string m_gallery_directory;
  float m_inference_match_threshold{0.363};
  std::chrono::milliseconds m_inference_interval{100};
  std::chrono::time_point<std::chrono::steady_clock> m_last_inference_at;
  SFaceContext m_prev_sface_ctx;
  std::vector<Identity> m_gallery;

  // Helper: Returns false if gallery cannot be populated
  bool load_gallery();

  void load_identities_from_folder(const std::string &folder_path,
                                   double threshold, IdentityCategory category,
                                   cv::FaceDetectorYN &yunet);
};

} // namespace MatrixPipeline::ProcessingUnit