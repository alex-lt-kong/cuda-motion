#pragma once

#include <cstdint>
#include <optional>

namespace CudaMotion::ProcessingUnit {
struct PipelineContext {
  // false means it is a gray image
  bool captured_from_real_device = false;
  int64_t capture_timestamp_ms = 0;
  int64_t capture_from_this_device_since_ms = 0;
  uint32_t frame_seq_num = 0;
  int processing_unit_idx = 0;
  float change_rate = -1;
  float fps = 0.0;
};
}