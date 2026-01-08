#pragma once

#include <opencv2/opencv.hpp>

namespace MatrixPipeline::ProcessingUnit {

struct DeviceInfo {
  std::string name;
  std::string uri;
  cv::Size expected_frame_size;
};

struct FaceRecognitionResult {
  cv::Mat embedding;      // The 128-d vector
  std::string identity;   // "Unknown" or matched name
  float similarity_score; // Cosine distance
  int matched_idx;        // Index in your gallery (optional)
};

// Add this to your PipelineContext struct
struct SFaceContext {
  // 1-to-1 mapping with YuNetContext: yunet[i] corresponds to results[i]
  std::vector<FaceRecognitionResult> results;
  bool model_loaded = false;
};

struct FaceDetection {
  cv::Rect2f bbox;
  std::array<cv::Point2f, 5> landmarks; // 5 points: eyes, nose, mouth corners
  float confidence;
};
using YuNetContext = std::vector<FaceDetection>;

struct YoloContext {
  // std::vector<cv::Mat> inference_outputs;
  cv::Size inference_input_size;
  std::vector<cv::Rect> boxes;
  std::vector<size_t> class_ids;
  std::vector<short> is_detection_valid;
  std::vector<float> confidences;
  std::vector<int> indices;
};

struct PipelineContext {
  DeviceInfo device_info;
  // false means it is a gray image
  bool captured_from_real_device = false;
  std::chrono::time_point<std::chrono::steady_clock, std::chrono::milliseconds>
      capture_timestamp;
  std::chrono::time_point<std::chrono::steady_clock, std::chrono::milliseconds>
      capture_from_this_device_since;
  uint32_t frame_seq_num = 0;
  size_t processing_unit_idx = 0;
  float change_rate = -1;
  float fps = 0.0;
  std::chrono::steady_clock::time_point latency_start_time;

  YoloContext yolo;
  YuNetContext yunet;
  SFaceContext sface;
};
} // namespace MatrixPipeline::ProcessingUnit