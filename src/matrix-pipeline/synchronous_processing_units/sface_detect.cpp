#include "sface_detect.h"

#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
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

    // 4. Load Gallery
    if (!load_gallery()) {
      SPDLOG_ERROR("Gallery loading failed. Aborting init.");
      return false;
    }
    m_inference_interval = std::chrono::milliseconds(
        config.value("inferenceIntervalMs", m_inference_interval.count()));
    SPDLOG_INFO("gallery.size(): {}, inference_interval: {}(ms)",
                m_gallery.size(), m_inference_interval.count());
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
    SPDLOG_ERROR("!fs::exists({})", m_gallery_directory);
    return false;
  }

  // 1. Temporary YuNet detector for gallery processing
  // We use a high confidence threshold here to ensure the "Gold Standard" for
  // our gallery
  cv::Ptr<cv::FaceDetectorYN> gallery_detector =
      cv::FaceDetectorYN::create(m_model_path_yunet, "", cv::Size(0, 0), 0.5f);

  for (const auto &entry : fs::directory_iterator(m_gallery_directory)) {
    if (!entry.is_directory())
      continue;

    std::string identity = entry.path().filename().string();
    cv::Mat aggregated_feature = cv::Mat::zeros(1, 128, CV_32F);
    int valid_samples_count = 0;

    SPDLOG_INFO("Loading gallery for identity: {}", identity);

    for (const auto &img_entry : fs::directory_iterator(entry.path())) {
      if (!img_entry.is_regular_file())
        continue;

      cv::Mat img = cv::imread(img_entry.path().string());
      if (img.empty())
        continue;

      gallery_detector->setInputSize(img.size());
      cv::Mat faces;
      gallery_detector->detect(img, faces);

      if (faces.rows < 1) {
        SPDLOG_WARN("No face detected in {}, skipping.",
                    img_entry.path().filename().string());
        continue;
      }

      // Pro-Tip: Only use samples where the detector is extremely certain.
      // Landmarks must be precise for the SFace embedding to be stable.
      float confidence = faces.at<float>(0, 14);
      if (constexpr float YUNET_CONF_THRESHOLD = 0.95f;
          confidence < YUNET_CONF_THRESHOLD) {
        SPDLOG_WARN("Skipping {} - landmark confidence too low ({:.2f})",
                    img_entry.path().filename().string(), confidence);
        continue;
      }

      cv::Mat aligned_face, feature_embedding;
      m_sface->alignCrop(img, faces.row(0), aligned_face);
      m_sface->feature(aligned_face, feature_embedding);

      // Accumulate the vectors
      aggregated_feature += feature_embedding;
      valid_samples_count++;
    }

    // 2. Finalize the Identity Centroid
    if (valid_samples_count > 0) {
      // Calculate the mean
      aggregated_feature /= static_cast<float>(valid_samples_count);

      // 3. Normalization: The "Magic" Step
      // SFace match() expects vectors to be on the unit hypersphere (length
      // = 1.0). Averaging pulls the vector "inside" the sphere; this puts it
      // back on the surface.
      cv::normalize(aggregated_feature, aggregated_feature, 1, 0, cv::NORM_L2);

      m_gallery.emplace_back(identity, aggregated_feature.clone());
      SPDLOG_INFO("Identity '{}' loaded. Samples used: {}", identity,
                  valid_samples_count);
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
  if (ctx.yunet.empty()) {
    return failure_and_continue;
  }

  // 3. Download Frame
  cv::Mat frame_cpu;
  try {
    frame.download(frame_cpu);
  } catch (const cv::Exception &e) {
    SPDLOG_ERROR("GPU download failed: {}", e.what());
    return success_and_continue;
  }

  // 4. Process Detections
  for (const auto &detection : ctx.yunet) {
    FaceRecognitionResult result;
    result.identity = "Unknown";
    result.similarity_score = 0.0f;
    result.matched_idx = -1;

    cv::Mat aligned_face;

    m_sface->alignCrop(frame_cpu, detection.face, aligned_face);

    if (!aligned_face.empty()) {
      cv::Mat feature_embedding;
      m_sface->feature(aligned_face, feature_embedding);
      result.embedding = feature_embedding.clone();

      double max_score = 0.0;
      int best_idx = -1;

      for (size_t i = 0; i < m_gallery.size(); ++i) {
        double score = m_sface->match(feature_embedding, m_gallery[i].second,
                                      cv::FaceRecognizerSF::DisType::FR_COSINE);

        if (score > max_score) {
          max_score = score;
          best_idx = static_cast<int>(i);
        }
      }

      if (max_score > m_match_threshold && best_idx != -1) {
        result.similarity_score = static_cast<float>(max_score);
        result.matched_idx = best_idx;
        result.identity = m_gallery[best_idx].first;
      }
    }

    ctx.sface.results.push_back(result);
  }

  m_prev_sface_ctx = ctx.sface;
  return success_and_continue;
}

} // namespace MatrixPipeline::ProcessingUnit