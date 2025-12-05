#pragma once

#include "../interfaces/i_synchronous_processing_unit.h"

#include <nlohmann/json.hpp>
#include <opencv2/cudawarping.hpp> // Required for cv::cuda::resize
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <string>

using njson = nlohmann::json;

namespace CudaMotion::ProcessingUnit {

class ResizeFrame final : public ISynchronousProcessingUnit {
private:
  int m_target_width{0};
  int m_target_height{0};
  double m_scale_factor{0.0};
  int m_interpolation{cv::INTER_LINEAR};

public:
  inline ResizeFrame() = default;
  inline ~ResizeFrame() override = default;

  /**
   * @brief Initializes resize parameters from JSON.
   * * Supported Config Modes:
   * 1. Absolute: { "width": 1920, "height": 1080 }
   * 2. Scaling:  { "scale": 0.5 }
   * * Optional: { "interpolation": "nearest" | "linear" | "cubic" | "area" }
   * Default interpolation is Linear.
   */
  bool init(const njson &config) override {
    try {
      // 1. Parse Dimensions
      if (config.contains("width")) m_target_width = config["width"].get<int>();
      if (config.contains("height")) m_target_height = config["height"].get<int>();
      if (config.contains("scale")) m_scale_factor = config["scale"].get<double>();

      // 2. Parse Interpolation Method
      if (config.contains("interpolation")) {
        std::string algo = config["interpolation"].get<std::string>();
        if (algo == "nearest") m_interpolation = cv::INTER_NEAREST;
        else if (algo == "linear") m_interpolation = cv::INTER_LINEAR;
        else if (algo == "cubic") m_interpolation = cv::INTER_CUBIC;
        else if (algo == "area") m_interpolation = cv::INTER_AREA;
        else {
            SPDLOG_WARN("Unknown interpolation '{}', defaulting to Linear", algo);
        }
      }

      // Validation: Must have either target dimensions OR scale factor
      bool has_dims = (m_target_width > 0 && m_target_height > 0);
      bool has_scale = (m_scale_factor > 0.0);

      if (!has_dims && !has_scale) {
        SPDLOG_ERROR("ResizeFrame: Config must provide 'width'/'height' OR 'scale'");
        return false;
      }

      return true;
    } catch (const std::exception& e) {
      SPDLOG_ERROR("ResizeFrame Init Error: {}", e.what());
      return false;
    }
  }

  [[nodiscard]] SynchronousProcessingResult process(cv::cuda::GpuMat &frame, [[maybe_unused]]ProcessingMetaData& meta_data) override {
    if (frame.empty()) return failure_and_continue;

    cv::Size new_size;

    // Determine target size
    if (m_target_width > 0 && m_target_height > 0) {
        new_size = cv::Size(m_target_width, m_target_height);
    } else if (m_scale_factor > 0.0) {
        new_size = cv::Size(static_cast<int>(frame.cols * m_scale_factor), 
                            static_cast<int>(frame.rows * m_scale_factor));
    } else {
        // configuration failed logic, generally shouldn't reach here if init checks passed
        return failure_and_continue; 
    }

    // Sanity check
    if (new_size.width <= 0 || new_size.height <= 0) return failure_and_continue;

    try {
        // cv::cuda::resize requires the destination to be allocated.
        // If we want to replace 'frame', we typically resize to a temp buffer 
        // and then move/assign it back.
        cv::cuda::GpuMat resized;
        cv::cuda::resize(frame, resized, new_size, 0, 0, m_interpolation);

        // Move the resized buffer into the pipeline frame
        frame = std::move(resized);

        return success_and_continue;
    } catch (const cv::Exception &e) {
        SPDLOG_ERROR("OpenCV CUDA Resize Error: {}", e.what());
        return failure_and_continue;
    }
  }
};

} // namespace CudaMotion::ProcessingUnit