#pragma once

#include <nvjpeg.h>
#include <opencv2/cudaimgproc.hpp>

#include <vector>

namespace CudaMotion::Utils{

class NvJpegEncoder {
public:
  NvJpegEncoder();

  ~NvJpegEncoder();

  bool encode(const cv::cuda::GpuMat &src, std::vector<uchar> &output_buffer, int quality = 90) const;

private:
  nvjpegHandle_t m_handle = nullptr;
  nvjpegEncoderState_t m_state = nullptr;
  nvjpegEncoderParams_t m_params = nullptr;

  static void check(nvjpegStatus_t status, const char *msg);
};
}