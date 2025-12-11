#pragma once

#include "../interfaces/i_asynchronous_processing_unit.h" // Assuming your base class is here
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

#include <string>

namespace MatrixPipeline::ProcessingUnit{

using njson = nlohmann::json;

class RtspProducer : public IAsynchronousProcessingUnit {
public:
  RtspProducer();
  virtual ~RtspProducer();

  /**
   * @brief Initializes the GStreamer pipeline and connects to MediaMTX.
   * Expected config JSON:
   * {
   * "rtsp_url": "rtsp://127.0.0.1:8554/mystream",
   * "width": 1920,
   * "height": 1080,
   * "fps": 30,
   * "bitrate_kbps": 4000
   * }
   */
  bool init(const njson &config) override;

protected:
  /**
   * @brief Called by the worker thread. Downloads GPU frame and pushes to RTSP.
   */
  void on_frame_ready(cv::cuda::GpuMat &frame, PipelineContext &ctx) override;

private:
  /**
   * @brief Helper to build the GStreamer pipeline string.
   */
  std::string build_pipeline(int width, int height, int fps, int bitrate_kbps, const std::string& url);

  cv::VideoWriter m_writer;

  // Cache config for potential reconnection logic (optional)
  std::string m_pipeline_string;
  cv::Size m_frame_size;
  double m_fps;
};

}