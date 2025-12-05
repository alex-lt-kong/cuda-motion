#pragma once

#include <drogon/drogon.h>
#include <drogon/utils/Utilities.h>
#include <opencv2/cudaimgproc.hpp>
#include <nvjpeg.h>

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <spdlog/spdlog.h>
#include <string>
#include <thread>
#include <vector>
#include <list> // Added for streaming clients list


namespace CudaMotion::Utils {
typedef auto(*signal_handler_callback)(int) -> void;

void execExternalProgramAsync(std::mutex &mtx, const std::string cmd,
                              const std::string &deviceName);

std::string getCurrentTimestamp() noexcept;

void install_signal_handler(signal_handler_callback cb);

/**
 * @brief Helper class to manage nvJPEG resources (Same as previous steps).
 */
class NvJpegEncoder {
public:
  NvJpegEncoder() {
    check(nvjpegCreateSimple(&m_handle), "CreateSimple");
    check(nvjpegEncoderStateCreate(m_handle, &m_state, NULL), "StateCreate");
    check(nvjpegEncoderParamsCreate(m_handle, &m_params, NULL), "ParamsCreate");
  }

  ~NvJpegEncoder() {
    if (m_params) nvjpegEncoderParamsDestroy(m_params);
    if (m_state) nvjpegEncoderStateDestroy(m_state);
    if (m_handle) nvjpegDestroy(m_handle);
  }

  bool encode(const cv::cuda::GpuMat &src, std::vector<uchar> &output_buffer, int quality = 90) {
    if (src.empty()) return false;

    check(nvjpegEncoderParamsSetSamplingFactors(m_params, NVJPEG_CSS_444, NULL), "SetSampling");
    check(nvjpegEncoderParamsSetQuality(m_params, quality, NULL), "SetQuality");

    nvjpegImage_t img_desc;
    img_desc.channel[0] = src.data;
    img_desc.pitch[0] = (unsigned int)src.step;

    nvjpegStatus_t status = nvjpegEncodeImage(m_handle, m_state, m_params,
                                              &img_desc, NVJPEG_INPUT_BGRI,
                                              src.cols, src.rows, NULL);

    if (status != NVJPEG_STATUS_SUCCESS) {
      SPDLOG_ERROR("nvjpegEncodeImage failed: {}", (int)status);
      return false;
    }

    size_t length;
    check(nvjpegEncodeRetrieveBitstream(m_handle, m_state, NULL, &length, NULL), "RetrieveLength");
    output_buffer.resize(length);
    check(nvjpegEncodeRetrieveBitstream(m_handle, m_state, output_buffer.data(), &length, NULL), "RetrieveData");
    return true;
  }

private:
  nvjpegHandle_t m_handle = nullptr;
  nvjpegEncoderState_t m_state = nullptr;
  nvjpegEncoderParams_t m_params = nullptr;

  void check(nvjpegStatus_t status, const char *msg) {
    if (status != NVJPEG_STATUS_SUCCESS) {
      SPDLOG_ERROR("NvJpeg Error [{}]: {}", msg, (int)status);
    }
  }
};
}