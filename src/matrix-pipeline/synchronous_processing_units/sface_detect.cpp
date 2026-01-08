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

SfaceDetect::SfaceDetect(const std::string &unit_path)
    : ISynchronousProcessingUnit(unit_path), m_match_threshold(0.363f) {
  // Constructor is now strictly empty of path configuration.
}

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

    SPDLOG_INFO("Initialization successful. Gallery size: {}",
                m_gallery.size());
    return true;

  } catch (const cv::Exception &e) {
    SPDLOG_ERROR("OpenCV Exception during init: {}", e.what());
    return false;
  } catch (const std::exception &e) {
    SPDLOG_ERROR("Std Exception during init: {}", e.what());
    return false;
  } catch (...) {
    SPDLOG_ERROR("Unknown exception during init.");
    return false;
  }
}

bool SfaceDetect::load_gallery() {
  if (!fs::exists(m_gallery_directory)) {
    SPDLOG_ERROR("Gallery folder not found: {}", m_gallery_directory);
    return false;
  }

  SPDLOG_INFO("Scanning gallery folder: {}", m_gallery_directory);

  cv::Ptr<cv::FaceDetectorYN> gallery_detector;

  try {
    if (!fs::exists(m_model_path_yunet)) {
      SPDLOG_ERROR("YuNet model for gallery processing "
                   "not found at: {}",
                   m_model_path_yunet);
      return false;
    }

    gallery_detector = cv::FaceDetectorYN::create(m_model_path_yunet, "",
                                                  cv::Size(0, 0), 0.5f);
  } catch (const std::exception &e) {
    SPDLOG_ERROR("Failed to create temp detector: {}", e.what());
    return false;
  }

  for (const auto &entry : fs::directory_iterator(m_gallery_directory)) {
    if (!entry.is_regular_file())
      continue;

    std::string filename = entry.path().filename().string();
    std::string identity = entry.path().stem().string();

    cv::Mat img = cv::imread(entry.path().string());
    if (img.empty()) {
      SPDLOG_WARN("Cannot read image: {}. Skipping.", filename);
      continue;
    }

    gallery_detector->setInputSize(img.size());
    cv::Mat faces;
    gallery_detector->detect(img, faces);

    if (faces.rows < 1) {
      SPDLOG_WARN("No face found in {}. Skipping.", filename);
      continue;
    }

    cv::Mat face = faces.row(0);
    std::vector<cv::Point2f> landmarks;
    for (int i = 0; i < 5; ++i) {
      float x = face.at<float>(0, 4 + 2 * i);
      float y = face.at<float>(0, 5 + 2 * i);
      landmarks.emplace_back(x, y);
    }

    cv::Mat aligned_face;
    m_sface->alignCrop(img, landmarks, aligned_face);

    if (aligned_face.empty()) {
      SPDLOG_WARN("Alignment failed for {}. Skipping.", filename);
      continue;
    }

    cv::Mat feature_emb;
    m_sface->feature(aligned_face, feature_emb);

    m_gallery.emplace_back(identity, feature_emb.clone());
    SPDLOG_INFO("Loaded identity: {}", identity);
  }

  if (m_gallery.empty()) {
    SPDLOG_ERROR("Gallery is empty! Please add images to {}",
                 m_gallery_directory);
    return false;
  }

  return true;
}

SynchronousProcessingResult SfaceDetect::process(cv::cuda::GpuMat &frame,
                                                 PipelineContext &ctx) {
  // 1. Reset Context
  ctx.sface.results.clear();

  // 2. Fast Exit
  if (ctx.yunet.empty()) {
    return success_and_continue;
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

    m_sface->alignCrop(frame_cpu, detection.landmarks, aligned_face);

    if (!aligned_face.empty()) {
      cv::Mat feature_emb;
      m_sface->feature(aligned_face, feature_emb);
      result.embedding = feature_emb.clone();

      double max_score = 0.0;
      int best_idx = -1;

      for (size_t i = 0; i < m_gallery.size(); ++i) {
        double score = m_sface->match(feature_emb, m_gallery[i].second,
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
    SPDLOG_INFO("ctx.sface.results.push_back({})", result.identity);
  }

  return success_and_continue;
}

} // namespace MatrixPipeline::ProcessingUnit