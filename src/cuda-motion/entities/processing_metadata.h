#pragma once

namespace CudaMotion::ProcessingUnit {
struct ProcessingMetaData {
  int64_t capture_timestamp_ms = 0;
  uint32_t seq_num = 0;
  float change_rate = -1;
  float fps = 0.0;
};
}