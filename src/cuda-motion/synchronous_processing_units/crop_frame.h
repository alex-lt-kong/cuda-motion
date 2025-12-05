#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

#include <nlohmann/json.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>

using njson = nlohmann::json;
namespace CudaMotion::ProcessingUnit {

class CropFrame final : public ISynchronousProcessingUnit {
private:
  float m_cropLeft{0.0f};
  float m_cropRight{0.0f};
  float m_cropTop{0.0f};
  float m_cropBottom{0.0f};

public:
  inline CropFrame() = default;
  inline ~CropFrame() override = default;

  /**
   * @brief Initializes cropping parameters from JSON.
   * Expected JSON: { "left": 0.1, "right": 0.1, "top": 0.0, "bottom": 0.0 }
   * Values are percentages of the total width/height to remove from that side.
   */
  bool init(const njson &config) override {
    try {
      if (config.contains("left")) m_cropLeft = config["left"].get<float>();
      if (config.contains("right")) m_cropRight = config["right"].get<float>();
      if (config.contains("top")) m_cropTop = config["top"].get<float>();
      if (config.contains("bottom")) m_cropBottom = config["bottom"].get<float>();

      // Basic validation to prevent invalid ROI (cropping 100% or more)
      if (m_cropLeft + m_cropRight >= 1.0f || m_cropTop + m_cropBottom >= 1.0f) {
        return false;
      }
      return true;
    } catch (...) {
      return false;
    }
  }

  [[nodiscard]] SynchronousProcessingResult process(cv::cuda::GpuMat &frame, [[maybe_unused]]ProcessingMetaData& meta_data) override {
    if (frame.empty()) return failure_and_continue;

    // Calculate crop dimensions
    int x = static_cast<int>(frame.cols * m_cropLeft);
    int y = static_cast<int>(frame.rows * m_cropTop);
    int width = frame.cols - x - static_cast<int>(frame.cols * m_cropRight);
    int height = frame.rows - y - static_cast<int>(frame.rows * m_cropBottom);

    // Sanity check dimensions
    if (width <= 0 || height <= 0) return failure_and_continue;

    // Apply Crop using ROI
    // This updates the GpuMat header to point only to the internal sub-rectangle.
    // It is an O(1) operation and does not copy deep memory.
    cv::Rect roi(x, y, width, height);
    frame = frame(roi);

    return success_and_continue;
  }
};

}