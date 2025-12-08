#pragma once

#include <drogon/drogon.h>
#include <drogon/utils/Utilities.h>
#include <opencv2/cudaimgproc.hpp>
#include <nvjpeg.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <cpr/cpr.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>


namespace CudaMotion::Utils {
using njson = nlohmann::json;
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

class MatrixSender {
private:
  std::string homeServer;
  std::string accessToken;
  std::string roomId;

  static void stbiWriteFunc(void *context, void *data, int size) {
    auto *buffer = static_cast<std::vector<unsigned char>*>(context);
    const unsigned char* bytes = static_cast<const unsigned char*>(data);
    buffer->insert(buffer->end(), bytes, bytes + size);
  }

  std::string upload(const std::string& data, const std::string& contentType) {
    std::string url = homeServer + "/_matrix/media/r0/upload";

    cpr::Response r = cpr::Post(
        cpr::Url{url},
        cpr::Header{
            {"Authorization", "Bearer " + accessToken},
            {"Content-Type", contentType}
        },
        cpr::Body{data}
    );

    if (r.status_code != 200) {
      SPDLOG_ERROR("Upload failed ({}: {})", r.status_code, r.text);
      return "";
    }

    try {
      auto j = njson::parse(r.text);
      return j["content_uri"];
    } catch (...) {
      return "";
    }
  }

  void sendEvent(const njson& contentBody, const std::string& msgType) {
    auto now = std::chrono::system_clock::now().time_since_epoch().count();
    std::string txnId = std::to_string(now);

    std::string url = homeServer + "/_matrix/client/r0/rooms/" + roomId +
                      "/send/m.room.message/" + txnId;

    njson payload;
    payload["msgtype"] = msgType;
    payload["body"] = contentBody["body"];
    payload["url"] = contentBody["url"];

    if (contentBody.contains("info")) {
      payload["info"] = contentBody["info"];
    }

    cpr::Response r = cpr::Put(
        cpr::Url{url},
        cpr::Header{
            {"Authorization", "Bearer " + accessToken},
            {"Content-Type", "application/njson"}
        },
        cpr::Body{payload.dump()}
    );

    if (r.status_code != 200) {
      std::cerr << "[Matrix] Send message failed: " << r.text << std::endl;
    } else {
      SPDLOG_INFO("sent {} successfully", msgType);
    }
  }

  std::string readFile(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) return "";
    auto pos = ifs.tellg();
    std::string result(pos, '\0');
    ifs.seekg(0, std::ios::beg);
    ifs.read(&result[0], pos);
    return result;
  }

  // Helper to clean URL strings (remove trailing slash)
  static std::string sanitizeUrl(std::string url) {
    while (!url.empty() && url.back() == '/') {
      url.pop_back();
    }
    return url;
  }

public:
  // Constructor now ONLY takes the strings. No Env vars here.
  MatrixSender(std::string url, std::string token, std::string room) {
    if (url.empty() || token.empty() || room.empty()) {
      throw std::runtime_error("MatrixSender requires URL, Token, and RoomID");
    }
    homeServer = sanitizeUrl(url);
    accessToken = token;
    roomId = room;
  }

  void sendText(const std::string& message) {
    if (message.empty()) return;
    njson content;
    content["body"] = message;
    sendEvent(content, "m.text");
  }

  void send_jpeg(const std::vector<uchar>& jpeg_bytes, int width, int height, const std::string& caption = "Image") {
    // TODO: unnecessary copy
    const std::string data(jpeg_bytes.begin(), jpeg_bytes.end());

    if (std::string mxc = upload(data, "image/jpeg"); !mxc.empty()) {
      njson content;
      content["body"] = caption;
      content["url"] = mxc;
      content["info"]["w"] = width;
      content["info"]["h"] = height;
      content["info"]["mimetype"] = "image/jpeg";
      content["info"]["size"] = data.size();
      sendEvent(content, "m.image");
    }
  }
/*
  void sendGif(const std::string& filepath, const std::string& caption = "GIF Clip") {
    std::string data = readFile(filepath);
    if (data.empty()) return;

    std::string mxc = upload(data, "image/gif");
    if (!mxc.empty()) {
      njson content;
      content["body"] = caption;
      content["url"] = mxc;
      int w, h, comp;
      if (stbi_info_from_memory(reinterpret_cast<const unsigned char*>(data.data()),
                                static_cast<int>(data.size()), &w, &h, &comp)) {
        content["info"]["w"] = w;
        content["info"]["h"] = h;
                                }
      content["info"]["mimetype"] = "image/gif";
      content["info"]["size"] = data.size();
      sendEvent(content, "m.image");
    }
  }*/

  void sendVideo(const std::string& filepath, const std::string& caption, int duration_ms = 0) {
    std::string data = readFile(filepath);
    if (data.empty()) return;
    std::string mxc = upload(data, "video/mp4");
    if (!mxc.empty()) {
      njson content;
      content["body"] = caption;
      content["url"] = mxc;
      njson info;
      info["mimetype"] = "video/mp4";
      info["size"] = data.size();
      if (duration_ms > 0) info["duration"] = duration_ms;
      content["info"] = info;
      sendEvent(content, "m.video");
    }
  }
};
}