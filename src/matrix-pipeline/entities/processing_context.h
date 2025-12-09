#pragma once

#include <opencv2/opencv.hpp>

#include <cstdint>
#include <algorithm>

namespace CudaMotion::ProcessingUnit {

struct YoloContext {
  std::vector<cv::Mat> inference_outputs;
  cv::Size inference_input_size;
  std::vector<cv::Rect> boxes;
  std::vector<size_t> class_ids;
  std::vector<float> confidences;
  std::vector<int> indices;
};

struct PipelineContext {
  // false means it is a gray image
  bool captured_from_real_device = false;
  int64_t capture_timestamp_ms = 0;
  int64_t capture_from_this_device_since_ms = 0;
  uint32_t frame_seq_num = 0;
  size_t processing_unit_idx = 0;
  float change_rate = -1;
  float fps = 0.0;

  YoloContext yolo;
};
} // namespace CudaMotion::ProcessingUnit