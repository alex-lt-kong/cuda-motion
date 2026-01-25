#include "sface_detect.h"

#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

#include <filesystem>

namespace fs = std::filesystem;

namespace MatrixPipeline::ProcessingUnit {

bool SfaceDetect::init(const nlohmann::json &config) {

  try {

    if (const auto key = "modelPathSface"; config.contains(key)) {
      m_model_path_sface = config[key].get<std::string>();
    } else {
      SPDLOG_ERROR("{} undefined", key);
      return false;
    }
    if (!fs::exists(m_model_path_sface)) {
      SPDLOG_ERROR("SFace model not found at: {}", m_model_path_sface);
      return false;
    }

    if (const auto key = "modelPathYunet"; config.contains(key)) {
      m_model_path_yunet = config[key].get<std::string>();
    } else {
      SPDLOG_ERROR("{} undefined", key);
      return false;
    }
    if (const auto key = "galleryDirectory"; config.contains(key)) {
      m_gallery_directory = config[key].get<std::string>();
    } else {
      SPDLOG_ERROR("{} undefined", key);
      return false;
    }

    m_authorized_enrollment_face_score_threshold =
        config.value("authorizedEnrollmentFaceScoreThreshold",
                     m_authorized_enrollment_face_score_threshold);
    m_unauthorized_enrollment_face_score_threshold =
        config.value("unauthorizedEnrollmentFaceScoreThreshold",
                     m_unauthorized_enrollment_face_score_threshold);
    m_inference_face_score_threshold = config.value(
        "inferenceFaceScoreThreshold", m_inference_face_score_threshold);
    m_inference_match_threshold =
        config.value("inferenceMatchThreshold", m_inference_match_threshold);

    SPDLOG_INFO("Loading SFace model...");
    m_sface = cv::FaceRecognizerSF::create(m_model_path_sface, "",
                                           cv::dnn::DNN_BACKEND_CUDA,
                                           cv::dnn::DNN_TARGET_CUDA);

    if (m_sface.empty()) {
      SPDLOG_ERROR("Failed to create SFace model instance.");
      return false;
    }

    if (!load_gallery()) {
      SPDLOG_ERROR("load_gallery() failed");
      return false;
    }

    m_inference_interval = std::chrono::milliseconds(
        config.value("inferenceIntervalMs", m_inference_interval.count()));

    SPDLOG_INFO(
        "gallery.size(): {}, inference_interval: {}ms, "
        "authorized_enrollment_face_score_threshold: {}, "
        "unauthorized_enrollment_face_score_threshold: {}, "
        "inference_face_score_threshold: {}, m_inference_match_threshold: {}",
        m_gallery.size(), m_inference_interval.count(),
        m_authorized_enrollment_face_score_threshold,
        m_unauthorized_enrollment_face_score_threshold,
        m_inference_face_score_threshold.has_value()
            ? std::to_string(m_inference_face_score_threshold.value())
            : "nullopt",
        m_inference_match_threshold);

    return true;

  } catch (const std::exception &e) {
    SPDLOG_ERROR("Error: {}", e.what());
    return false;
  }
}

bool SfaceDetect::load_gallery() {
  if (!fs::exists(m_gallery_directory)) {
    SPDLOG_ERROR("gallery_directory does not exist: {}", m_gallery_directory);
    return false;
  }

  // Initialize YuNet ONCE for the loading process.
  // We set the internal threshold low (0.3) so it detects almost everything,
  // allowing us to filter manually with specific thresholds in the helper loop.
  cv::Ptr<cv::FaceDetectorYN> yunet =
      cv::FaceDetectorYN::create(m_model_path_yunet, "", cv::Size(0, 0), 0.3f);

  fs::path root(m_gallery_directory);
  fs::path authorized_path = root / "authorized";
  fs::path unauthorized_path = root / "unauthorized";

  bool loaded_something = false;

  // 1. Load Authorized (High strictness)
  if (fs::exists(authorized_path)) {
    SPDLOG_INFO("Loading Authorized identities from: {}",
                authorized_path.string());
    load_identities_from_folder(authorized_path.string(),
                                m_authorized_enrollment_face_score_threshold,
                                IdentityCategory::Authorized, *yunet);
    loaded_something = true;
  }

  // 2. Load Unauthorized (Low strictness for CCTV)
  if (fs::exists(unauthorized_path)) {
    SPDLOG_INFO("Loading Unauthorized identities from: {}",
                unauthorized_path.string());
    load_identities_from_folder(unauthorized_path.string(),
                                m_unauthorized_enrollment_face_score_threshold,
                                IdentityCategory::Unauthorized, *yunet);
    loaded_something = true;
  }

  // Fallback: If neither subfolder exists, try loading root as "Authorized" for
  // backward compatibility
  if (!loaded_something) {
    SPDLOG_WARN("No 'authorized' or 'unauthorized' subfolders found. Scanning "
                "root as Authorized.");
    load_identities_from_folder(m_gallery_directory,
                                m_authorized_enrollment_face_score_threshold,
                                IdentityCategory::Authorized, *yunet);
  }

  return !m_gallery.empty();
}

void SfaceDetect::load_identities_from_folder(const std::string &folder_path,
                                              double threshold,
                                              IdentityCategory category,
                                              cv::FaceDetectorYN &yunet) {

  for (const auto &entry : fs::directory_iterator(folder_path)) {
    if (!entry.is_directory())
      continue;

    Identity identity;
    identity.name = entry.path().filename().string();
    identity.category = category; // Set the category

    for (const auto &img_entry : fs::directory_iterator(entry.path())) {
      if (!img_entry.is_regular_file())
        continue;

      cv::Mat img = cv::imread(img_entry.path().string());
      if (img.empty()) {
        SPDLOG_WARN("cv::imread failed for {}", img_entry.path().string());
        continue;
      }

      yunet.setInputSize(img.size());
      cv::Mat faces;
      yunet.detect(img, faces);

      if (faces.rows < 1)
        continue;

      // Extract confidence from YuNet result (index 14)
      float confidence = faces.at<float>(0, 14);

      // Check against the specific threshold for this category
      if (confidence < threshold) {
        SPDLOG_WARN("Skipped {} (Score: {:.2f} < Threshold: {:.2f})",
                    img_entry.path().filename().string(), confidence,
                    threshold);

        // Rename logic (omitted for brevity, but you can keep it here)
        continue;
      }

      cv::Mat aligned_face, feature_embedding;
      m_sface->alignCrop(img, faces.row(0), aligned_face);
      m_sface->feature(aligned_face, feature_embedding);

      cv::Mat normalized_embedding;
      cv::normalize(feature_embedding, normalized_embedding, 1, 0, cv::NORM_L2);
      identity.normalized_embeddings.push_back(normalized_embedding.clone());
    }

    if (!identity.normalized_embeddings.empty()) {
      m_gallery.push_back(std::move(identity));
      SPDLOG_INFO(
          "Loaded '{}' ({}) with {} embeddings.", m_gallery.back().name,
          (category == IdentityCategory::Authorized ? "Auth" : "Unauth"),
          m_gallery.back().normalized_embeddings.size());
    }
  }
}

SynchronousProcessingResult SfaceDetect::process(cv::cuda::GpuMat &frame,
                                                 PipelineContext &ctx) {
  if (std::chrono::steady_clock::now() - m_last_inference_at <
      m_inference_interval) {
    ctx.sface = m_prev_sface_ctx;
    return failure_and_continue;
  }
  m_last_inference_at = std::chrono::steady_clock::now();

  ctx.sface.results.clear();
  if (ctx.yunet.empty())
    return failure_and_continue;

  cv::Mat frame_cpu;
  try {
    frame.download(frame_cpu);
  } catch (const cv::Exception &e) {
    SPDLOG_ERROR("frame.download() failed: {}", e.what());
    disable();
    return success_and_continue;
  }

  for (const auto &detection : ctx.yunet) {
    // Basic confidence check for inference
    if (m_inference_face_score_threshold.has_value() &&
        detection.confidence < m_inference_face_score_threshold.value()) {
      continue;
    }

    FaceRecognitionResult result;
    result.identity = "Unknown";
    result.similarity_score = std::numeric_limits<float>::quiet_NaN();
    result.matched_idx = -1;
    result.category = IdentityCategory::Unknown; // Default

    cv::Mat aligned_face;
    m_sface->alignCrop(frame_cpu, detection.face, aligned_face);

    if (aligned_face.empty())
      continue;

    cv::Mat probe_embedding, normalized_probe_embedding;
    // Where the DNN actually runs
    m_sface->feature(aligned_face, probe_embedding);
    cv::normalize(probe_embedding, normalized_probe_embedding, 1, 0,
                  cv::NORM_L2);

    result.embedding = normalized_probe_embedding.clone();

    double best_score_overall = -1.0;
    int best_identity_idx = -1;

    // Match against ALL identities (mixed Authorized and Unauthorized)
    for (size_t i = 0; i < m_gallery.size(); ++i) {
      double best_score_for_this_person = -1.0;

      for (const auto &normalized_gallery_embedding :
           m_gallery[i].normalized_embeddings) {
        double score = m_sface->match(normalized_probe_embedding,
                                      normalized_gallery_embedding,
                                      cv::FaceRecognizerSF::DisType::FR_COSINE);
        best_score_for_this_person =
            std::max(best_score_for_this_person, score);
      }

      if (best_score_for_this_person > best_score_overall) {
        best_score_overall = best_score_for_this_person;
        best_identity_idx = static_cast<int>(i);
      }
    }

    if (best_score_overall > m_inference_match_threshold &&
        best_identity_idx != -1) {
      result.similarity_score = static_cast<float>(best_score_overall);
      result.matched_idx = best_identity_idx;

      // Fill the identity info
      const auto &matched_identity = m_gallery[best_identity_idx];
      result.identity = matched_identity.name;
      result.category = matched_identity.category; // Set the category
    }

    ctx.sface.results.push_back(result);
  }

  m_prev_sface_ctx = ctx.sface;
  return success_and_continue;
}

} // namespace MatrixPipeline::ProcessingUnit