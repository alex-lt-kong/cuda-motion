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
    if (config.contains("modelPathSface")) {
      m_model_path_sface = config["modelPathSface"].get<std::string>();
    } else {
      SPDLOG_ERROR("modelPathSface undefined");
      return false;
    }

    if (config.contains("modelPathYunet")) {
      m_model_path_yunet = config["modelPathYunet"].get<std::string>();
    } else {
      SPDLOG_ERROR("modelPathYunet undefined");
      return false;
    }
    m_enrollment_face_score_threshold = config.value(
        "enrollmentFaceScoreThreshold", m_enrollment_face_score_threshold);
    m_inference_face_score_threshold = config.value(
        "inferenceFaceScoreThreshold", m_inference_face_score_threshold);

    if (config.contains("galleryDirectory")) {
      m_gallery_directory = config["galleryDirectory"].get<std::string>();
    } else {
      SPDLOG_ERROR("galleryDirectory undefined");
      return false;
    }

    // 2. Validate Filesystem Prerequisites
    if (!fs::exists(m_model_path_sface)) {
      SPDLOG_ERROR("SFace model not found at: {}", m_model_path_sface);
      return false;
    }

    // 3. Initialize SFace Model (CUDA)
    SPDLOG_INFO("Loading SFace model...");
    m_sface = cv::FaceRecognizerSF::create(m_model_path_sface, "",
                                           cv::dnn::DNN_BACKEND_CUDA,
                                           cv::dnn::DNN_TARGET_CUDA);

    if (m_sface.empty()) {
      SPDLOG_ERROR("Failed to create SFace model instance (returned empty).");
      return false;
    }

    if (!load_gallery()) {
      SPDLOG_ERROR("load_gallery() failed");
      return false;
    }
    m_inference_interval = std::chrono::milliseconds(
        config.value("inferenceIntervalMs", m_inference_interval.count()));

    SPDLOG_INFO("gallery.size(): {}, inference_interval: {}(ms), "
                "enrollment_face_score_threshold: {}, "
                "inference_face_score_threshold: {}",
                m_gallery.size(), m_inference_interval.count(),
                m_enrollment_face_score_threshold,
                m_inference_face_score_threshold.has_value()
                    ? std::to_string(m_inference_face_score_threshold.value())
                    : std::string("NaN"));
    return true;

  } catch (const std::exception &e) {
    SPDLOG_ERROR("Error: {}", e.what());
    return false;
  } catch (...) {
    SPDLOG_ERROR("Unknown exception during init.");
    return false;
  }
}

bool SfaceDetect::load_gallery() {
  if (!fs::exists(m_gallery_directory)) {
    SPDLOG_ERROR("gallery_directory does not exist: {}", m_gallery_directory);
    return false;
  }

  // here we set score_threshold to  m_enrollment_face_score_threshold / 2 to
  // provide some warnings to user
  cv::Ptr<cv::FaceDetectorYN> yunet =
      cv::FaceDetectorYN::create(m_model_path_yunet, "", cv::Size(0, 0),
                                 m_enrollment_face_score_threshold / 2);

  for (const auto &entry : fs::directory_iterator(m_gallery_directory)) {
    if (!entry.is_directory()) {
      SPDLOG_WARN("Skipping non-directory entry: {}", entry.path().string());
      continue;
    }

    Identity identity;
    identity.name = entry.path().filename().string();
    SPDLOG_INFO("Loading gallery for identity: {}", identity.name);

    for (const auto &img_entry : fs::directory_iterator(entry.path())) {
      if (!img_entry.is_regular_file())
        continue;

      cv::Mat img = cv::imread(img_entry.path().string());
      if (img.empty()) {
        SPDLOG_ERROR("cv::imread({}) is empty", img_entry.path().string());
        continue;
      }

      yunet->setInputSize(img.size());
      cv::Mat faces;
      yunet->detect(img, faces);

      if (faces.rows < 1) {
        SPDLOG_WARN("No face detected in {}, skipping.",
                    img_entry.path().filename().string());
        continue;
      }

      // High precision check for gallery quality
      // faces.at<float>(0, 14) extract the 14th value from YuNet's results
      if (const auto confidence = faces.at<float>(0, 14);
          confidence < m_enrollment_face_score_threshold) {

        SPDLOG_WARN(" {} skipped due to low landmark confidence ({:.2f} vs "
                    "landmark_confidence_threshold: {})",
                    img_entry.path().filename().string(), confidence,
                    m_enrollment_face_score_threshold);

        const auto old_path = img_entry.path();
        if (old_path.extension() == ".bak")
          continue;
        auto new_path = img_entry.path();
        new_path += ".bak";
        fs::rename(old_path, new_path);
        SPDLOG_WARN("{} fs::rename()ed to {}", old_path.filename().string(),
                    new_path.filename().string());
        continue;
      }

      cv::Mat aligned_face, feature_embedding;
      m_sface->alignCrop(img, faces.row(0), aligned_face);
      m_sface->feature(aligned_face, feature_embedding);

      // Crucial: SFace match() works best on normalized vectors.
      // Since we aren't averaging anymore, we normalize each individual vector.
      cv::Mat normalized_embedding;
      cv::normalize(feature_embedding, normalized_embedding, 1, 0, cv::NORM_L2);

      identity.embeddings.push_back(normalized_embedding.clone());
    }

    if (!identity.embeddings.empty()) {
      m_gallery.push_back(std::move(identity));
      SPDLOG_INFO("Identity '{}' loaded with {} embeddings.",
                  m_gallery.back().name, m_gallery.back().embeddings.size());
    } else {
      SPDLOG_WARN("Identity '{}' has no embeddings. Skipping.", identity.name);
    }
  }
  return true;
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

  // GPU to CPU transfer for SFace (SFace inference is currently CPU-based in
  // OpenCV, even if the backend is CUDA, the preprocessing often hits CPU).
  cv::Mat frame_cpu;
  try {
    frame.download(frame_cpu);
  } catch (const cv::Exception &e) {
    SPDLOG_ERROR("frame.download(frame_cpu) failed: {}", e.what());
    disable();
    return success_and_continue;
  }

  for (const auto &detection : ctx.yunet) {
    if (m_inference_face_score_threshold.has_value() &&
        detection.confidence < m_inference_face_score_threshold.value()) {
      return failure_and_continue;
    }
    FaceRecognitionResult result;
    result.identity = "Unknown";
    result.similarity_score = std::numeric_limits<float>::quiet_NaN();
    result.matched_idx = -1;

    cv::Mat aligned_face;
    m_sface->alignCrop(frame_cpu, detection.face, aligned_face);

    if (aligned_face.empty())
      return success_and_continue;

    cv::Mat probe_embedding, normalized_probe;
    m_sface->feature(aligned_face, probe_embedding);
    cv::normalize(probe_embedding, normalized_probe, 1, 0, cv::NORM_L2);

    result.embedding = normalized_probe.clone();

    double best_score_overall = -1.0;
    int best_identity_idx = -1;

    // Iterate through each person in the gallery
    for (size_t i = 0; i < m_gallery.size(); ++i) {
      double best_score_for_this_person = -1.0;

      // Compare probe against EVERY embedding for this specific person
      for (const auto &stored_vec : m_gallery[i].embeddings) {
        double score = m_sface->match(normalized_probe, stored_vec,
                                      cv::FaceRecognizerSF::DisType::FR_COSINE);

        best_score_for_this_person =
            std::max(best_score_for_this_person, score);
      }

      if (best_score_for_this_person > best_score_overall) {
        best_score_overall = best_score_for_this_person;
        best_identity_idx = static_cast<int>(i);
      }
    }

    // Note: Since you have more vectors, consider raising m_match_threshold
    // by ~0.05
    if (best_score_overall > m_match_threshold && best_identity_idx != -1) {
      result.similarity_score = static_cast<float>(best_score_overall);
      result.matched_idx = best_identity_idx;
      result.identity = m_gallery[best_identity_idx].name;
    }

    ctx.sface.results.push_back(result);
  }

  m_prev_sface_ctx = ctx.sface;
  return success_and_continue;
}

} // namespace MatrixPipeline::ProcessingUnit