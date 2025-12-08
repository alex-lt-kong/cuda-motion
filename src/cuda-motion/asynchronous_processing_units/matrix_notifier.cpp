#include "../interfaces/i_asynchronous_processing_unit.h"
#include "../utils.h"


#include <nlohmann/json.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>

namespace CudaMotion {
namespace Utils {



}


namespace ProcessingUnit {
bool MatrixNotifier::init(const njson &config) {
  if (!config.contains("matrixHomeServer") || !config.contains("matrixRoomId") || !config.contains("matrixAccessToken")) {
    SPDLOG_ERROR("Missing matrix credentials");
    return false;
  }
  m_matrix_homeserver = config["matrixHomeServer"];
  m_matrix_room_id = config["matrixRoomId"];
  m_matrix_access_token = config["matrixAccessToken"];
  SPDLOG_INFO("matrix_homeserver: {}, matrix_room_id: {}, matrix_access_token: {}", m_matrix_homeserver, m_matrix_room_id, m_matrix_access_token);
  m_sender = std::make_unique<Utils::MatrixSender>(m_matrix_homeserver, m_matrix_access_token, m_matrix_room_id);
  m_gpu_encoder = std::make_unique<Utils::NvJpegEncoder>();
  return true;
}

void MatrixNotifier::on_frame_ready(cv::cuda::GpuMat &frame,
                                               [[maybe_unused]] PipelineContext &ctx) {
  if (ctx.frame_seq_num % m_notification_interval_frame != 0) return;
  if (ctx.yolo.indices.empty()) return;
  bool person_detected = false;
  for (const int idx : ctx.yolo.indices) {
    if (const int class_id = ctx.yolo.class_ids[idx]; class_id == 0) {
      person_detected = true;
      break;
    }
  }

  if (!person_detected) return;
  std::vector<uchar> jpeg_bytes;
  bool success = m_gpu_encoder->encode(frame, jpeg_bytes, 90);
  if (!success) {
    SPDLOG_ERROR("m_gpu_encoder->encode() failed");
    return;
  }
  m_sender->send_jpeg(jpeg_bytes, frame.cols, frame.rows, "Test Image");
}

};


}