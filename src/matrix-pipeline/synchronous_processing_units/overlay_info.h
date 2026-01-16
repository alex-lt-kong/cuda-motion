#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include <string>

namespace MatrixPipeline::ProcessingUnit {

class OverlayInfo final : public ISynchronousProcessingUnit {
public:
  explicit OverlayInfo(const std::string &unit_path)
      : ISynchronousProcessingUnit(unit_path + "/OverlayInfo") {}
  ~OverlayInfo() override = default;

  bool init(const nlohmann::json &config) override;

  [[nodiscard]] SynchronousProcessingResult
  process(cv::cuda::GpuMat &frame, PipelineContext &ctx) override;

private:
  void update_font_metrics(int frameRows);
  void uploadAndOverlay(cv::cuda::GpuMat &frame, cv::Rect roiRect);

  std::chrono::time_point<std::chrono::steady_clock> m_last_info_update_time;

  // --- Configuration ---
  std::string m_info_template;

  // --- Visual Settings ---
  float m_text_height_ratio{0.02f}; // 2% of frame height
  int m_margin_x{5};
  int m_margin_y{5};

  // Calculated per-frame
  float m_current_opencv_scale{1.0f};
  int m_current_thickness{2};
  int m_line_height_px{0};

  cv::Scalar m_text_color{255, 255, 255};
  cv::Scalar m_glow_color{2, 2, 2};

  // 0.0 = disable. 0.25 = border is 25% of font height.
  float m_outline_ratio{0.25f};
  int m_current_outline_thickness{0};

  // --- Buffers ---
  int m_stripHeight{0};
  cv::Mat m_h_text_strip;
  cv::cuda::GpuMat d_text_strip;
  cv::cuda::GpuMat d_strip_gray;
  cv::cuda::GpuMat d_mask;

  static constexpr float BASE_FONT_HEIGHT_PX = 22.0f;
};

} // namespace MatrixPipeline::ProcessingUnit