#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"
#include "yolo_detect.h"

#include <opencv2/cudaimgproc.hpp>

namespace MatrixPipeline::ProcessingUnit {

class YoloOverlayBoundingBoxes final : public ISynchronousProcessingUnit {
public:
  explicit YoloOverlayBoundingBoxes(const std::string &unit_path)
      : ISynchronousProcessingUnit(unit_path + "/YoloOverlayBoundingBoxes") {}
  ~YoloOverlayBoundingBoxes() override = default;

  bool init(const njson &config) override;

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx) override;

private:
  // --- State ---;
  float m_label_font_scale{0.5};
  std::vector<cv::Scalar> m_colors;
  std::vector<std::string> m_class_names = {
      "person",        "bicycle",      "car",
      "motorcycle",    "airplane",     "bus",
      "train",         "truck",        "boat",
      "traffic light", "fire hydrant", "stop sign",
      "parking meter", "bench",        "bird",
      "cat",           "dog",          "horse",
      "sheep",         "cow",          "elephant",
      "bear",          "zebra",        "giraffe",
      "backpack",      "umbrella",     "handbag",
      "tie",           "suitcase",     "frisbee",
      "skis",          "snowboard",    "sports ball",
      "kite",          "baseball bat", "baseball glove",
      "skateboard",    "surfboard",    "tennis racket",
      "bottle",        "wine glass",   "cup",
      "fork",          "knife",        "spoon",
      "bowl",          "banana",       "apple",
      "sandwich",      "orange",       "broccoli",
      "carrot",        "hot dog",      "pizza",
      "donut",         "cake",         "chair",
      "couch",         "potted plant", "bed",
      "dining table",  "toilet",       "tv",
      "laptop",        "mouse",        "remote",
      "keyboard",      "cell phone",   "microwave",
      "oven",          "toaster",      "sink",
      "refrigerator",  "book",         "clock",
      "vase",          "scissors",     "teddy bear",
      "hair drier",    "toothbrush"};

  cv::Size m_model_input_size = {640, 640}; // Default YOLO size
  // --- Reusable Buffers (Avoid re-allocation) ---
  cv::Mat m_h_overlay_canvas;          // Host (CPU) Canvas
  cv::cuda::GpuMat m_d_overlay_canvas; // Device (GPU) Canvas
  cv::cuda::GpuMat m_d_overlay_gray;   // Intermediate Gray for masking
  cv::cuda::GpuMat d_overlay_mask;     // Final Mask
  std::optional<BoundingBoxScaleParams> m_scaling_params{std::nullopt};
};

} // namespace MatrixPipeline::ProcessingUnit