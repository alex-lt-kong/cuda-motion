#include "nvjpeg_encoder.h"

#include <spdlog/spdlog.h>

namespace MatrixPipeline::Utils {
NvJpegEncoder::NvJpegEncoder() {
  check(nvjpegCreateSimple(&m_handle), "CreateSimple");
  check(nvjpegEncoderStateCreate(m_handle, &m_state, nullptr), "StateCreate");
  check(nvjpegEncoderParamsCreate(m_handle, &m_params, nullptr),
        "ParamsCreate");
}

NvJpegEncoder::~NvJpegEncoder() {
  if (m_params)
    nvjpegEncoderParamsDestroy(m_params);
  if (m_state)
    nvjpegEncoderStateDestroy(m_state);
  if (m_handle)
    nvjpegDestroy(m_handle);
}

void NvJpegEncoder::check(nvjpegStatus_t status, const char *msg) {
  if (status != NVJPEG_STATUS_SUCCESS) {
    SPDLOG_ERROR("NvJpeg Error [{}]: {}", msg, (int)status);
  }
}

bool NvJpegEncoder::encode(const cv::cuda::GpuMat &src,
                           std::string &output_buffer,
                           const int quality) const {
  if (src.empty())
    return false;

  check(nvjpegEncoderParamsSetSamplingFactors(m_params, NVJPEG_CSS_444, NULL),
        "SetSampling");
  check(nvjpegEncoderParamsSetQuality(m_params, quality, NULL), "SetQuality");

  nvjpegImage_t img_desc;
  img_desc.channel[0] = src.data;
  img_desc.pitch[0] = (unsigned int)src.step;

  nvjpegStatus_t status =
      nvjpegEncodeImage(m_handle, m_state, m_params, &img_desc,
                        NVJPEG_INPUT_BGRI, src.cols, src.rows, NULL);

  if (status != NVJPEG_STATUS_SUCCESS) {
    SPDLOG_ERROR("nvjpegEncodeImage failed: {}", (int)status);
    return false;
  }

  size_t length;
  check(nvjpegEncodeRetrieveBitstream(m_handle, m_state, NULL, &length, NULL),
        "RetrieveLength");
  output_buffer.resize(length);
  check(nvjpegEncodeRetrieveBitstream(
            m_handle, m_state,
            reinterpret_cast<unsigned char *>(output_buffer.data()), &length,
            nullptr),
        "RetrieveData");
  return true;
}
} // namespace MatrixPipeline::Utils
