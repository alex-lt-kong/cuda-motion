#pragma once

#include "../entities/processing_context.h"
#include "../entities/synchronous_processing_result.h"

#include <nlohmann/json.hpp>
#include <opencv2/cudawarping.hpp>

using njson = nlohmann::json;

namespace CudaMotion::ProcessingUnit {

class ISynchronousProcessingUnit {
public:
  virtual ~ISynchronousProcessingUnit() = default;

  virtual bool init(const njson &config) = 0;

  ///
  /// @param frame the frame to be processed
  /// @return
  virtual SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                              PipelineContext &ctx) = 0;
};

class RotateFrame final : public ISynchronousProcessingUnit {
private:
  int m_angle{0};

public:
  RotateFrame() = default;
  ~RotateFrame() override = default;

  SynchronousProcessingResult
  process(cv::cuda::GpuMat &frame,
          [[maybe_unused]] PipelineContext &meta_data) override;

  bool init(const njson &config) override;
};

class ResizeFrame final : public ISynchronousProcessingUnit {
private:
  int m_target_width{0};
  int m_target_height{0};
  double m_scale_factor{0.0};
  int m_interpolation{cv::INTER_LINEAR};

public:
  ResizeFrame() = default;
  ~ResizeFrame() override = default;

  /**
   * @brief Initializes resize parameters from JSON.
   * * Supported Config Modes:
   * 1. Absolute: { "width": 1920, "height": 1080 }
   * 2. Scaling:  { "scale": 0.5 }
   * * Optional: { "interpolation": "nearest" | "linear" | "cubic" | "area" }
   * Default interpolation is Linear.
   */
  bool init(const njson &config) override;

  [[nodiscard]] SynchronousProcessingResult
  process(cv::cuda::GpuMat &frame, PipelineContext &ctx) override;
};

class DetectObjects final : public ISynchronousProcessingUnit {
private:
  std::string m_model_path;
  cv::Size m_model_input_size = {640, 640}; // Default YOLO size
  cv::dnn::Net m_net;
  float m_conf_thres = 0.5f;
  float m_nms_thres = 0.45f;
  int m_frame_interval = 10;
  YoloContext m_prev_yolo_ctx;

  void post_process_yolo(const cv::cuda::GpuMat &frame,
                         PipelineContext &ctx) const;

public:
  bool init(const njson &config) override;

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx) override;
};

class OverlayBoundingBoxes final : public ISynchronousProcessingUnit {
public:
  OverlayBoundingBoxes() = default;
  ~OverlayBoundingBoxes() override = default;

  bool init(const njson &config) override;

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx) override;

private:
  // --- State ---
  std::vector<std::string> m_class_names;
  std::vector<cv::Scalar> m_colors;

  // --- Reusable Buffers (Avoid re-allocation) ---
  cv::Mat h_overlay_canvas;          // Host (CPU) Canvas
  cv::cuda::GpuMat d_overlay_canvas; // Device (GPU) Canvas
  cv::cuda::GpuMat d_overlay_gray;   // Intermediate Gray for masking
  cv::cuda::GpuMat d_overlay_mask;   // Final Mask
};

class DebugOutput final : public ISynchronousProcessingUnit {
public:
  DebugOutput() = default;
  ~DebugOutput() override = default;

  bool init(const njson &config) override;

  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx) override;
};

} // namespace CudaMotion::ProcessingUnit