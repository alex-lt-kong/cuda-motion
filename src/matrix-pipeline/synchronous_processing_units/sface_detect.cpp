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

    if (const auto key = "modelPath"; config.contains(key)) {
      m_model_path_sface = config[key].get<std::string>();
    } else {
      SPDLOG_ERROR("{} undefined", key);
      return false;
    }
    if (!fs::exists(m_model_path_sface)) {
      SPDLOG_ERROR("SFace model not found at: {}", m_model_path_sface);
      return false;
    }

    if (const auto key = njson::json_pointer("/yuNet/modelPath");
        config.contains(key)) {
      m_model_path_yunet = config.at(key).get<std::string>();
    } else {
      SPDLOG_ERROR("{} undefined", key.to_string());
      return false;
    }
    if (const auto key = "galleryDirectory"; config.contains(key)) {
      m_gallery_directory = config[key].get<std::string>();
    } else {
      SPDLOG_ERROR("{} undefined", key);
      return false;
    }
    if (!m_yunet.init(config["yuNet"])) {
      SPDLOG_ERROR("m_yunet.init() init failed");
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
        m_inference_face_score_threshold, m_inference_match_threshold);

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

      const auto confidence = faces.at<float>(0, 14);
      if (confidence < threshold) {
        SPDLOG_WARN("Skipped {} (Score: {:.2f} < Threshold: {:.2f})",
                    img_entry.path().filename().string(), confidence,
                    threshold);
        const auto &old_path = img_entry.path();
        if (old_path.extension() == ".bak")
          continue;
        auto new_path = img_entry.path();
        new_path += ".bak";
        fs::rename(old_path, new_path);
        SPDLOG_WARN("{} fs::rename()ed to {}", old_path.filename().string(),
                    new_path.filename().string());
        continue;
      }
      SPDLOG_INFO("Adding {} (Score: {:.2f} >= Threshold: {:.2f})",
                  img_entry.path().filename().string(), confidence, threshold);

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
    ctx.yunet_sface = m_prev_yunet_sface_ctx;
    return failure_and_continue;
  }

  m_last_inference_at = std::chrono::steady_clock::now();
  ctx.yunet_sface.results.clear();
  m_prev_yunet_sface_ctx.results.clear();

  if (const auto res = m_yunet.process(frame, ctx);
      res == success_and_stop || res == failure_and_stop) {
    return failure_and_continue;
  }

  if (ctx.yunet_sface.results.empty())
    return success_and_continue;

  try {
    // frame.download() handles m_pinned_mem_for_cpu_frame's allocation
    frame.download(m_pinned_mem_for_cpu_frame);
    // Map the pinned data to a standard cv::Mat for processing
    // This is zero-copy (just pointer arithmetic)
    m_frame_cpu = m_pinned_mem_for_cpu_frame.createMatHeader();
  } catch (const cv::Exception &e) {
    SPDLOG_ERROR("frame.download() failed: {}", e.what());
    disable();
    ctx.yunet_sface.results.clear();
    return success_and_continue;
  }

  for (auto &result : ctx.yunet_sface.results) {
    if (result.detection.confidence < m_inference_face_score_threshold) {
      // std::nullopt should have been set in YuNet, but anyways
      result.recognition = std::nullopt;
      continue;
    }

    SFaceRecognition recognition;
    recognition.identity = "";
    recognition.similarity_score = std::numeric_limits<float>::quiet_NaN();
    recognition.matched_idx = -1;
    recognition.category = IdentityCategory::Unknown;

    m_sface->alignCrop(m_frame_cpu, result.detection.yunet_output,
                       m_aligned_face);
    if (m_aligned_face.empty())
      continue;

    cv::Mat probe_embedding, normalized_probe_embedding;
    // Where the DNN actually runs
    m_sface->feature(m_aligned_face, probe_embedding);
    cv::normalize(probe_embedding, normalized_probe_embedding, 1, 0,
                  cv::NORM_L2);

    // Gemini 3 Pro suggests we to keep the clone()
    recognition.embedding = normalized_probe_embedding.clone();

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
      recognition.similarity_score = static_cast<float>(best_score_overall);
      recognition.matched_idx = best_identity_idx;

      const auto &matched_identity = m_gallery[best_identity_idx];
      recognition.identity = matched_identity.name;
      recognition.category = matched_identity.category;
    }

    result.recognition = recognition;
  }

  m_prev_yunet_sface_ctx = ctx.yunet_sface;
  return success_and_continue;
}

} // namespace MatrixPipeline::ProcessingUnit