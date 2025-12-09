#include "matrix_sender.h"

#include <cpr/cpr.h>
#include <spdlog/spdlog.h>
#include <chrono>
#include <fstream>
#include <stdexcept>

namespace CudaMotion::Utils {

// -----------------------------------------------------------------------------
// Private Helpers
// -----------------------------------------------------------------------------
/*
void MatrixSender::stbiWriteFunc(void *context, void *data, int size) {
    auto *buffer = static_cast<std::vector<unsigned char>*>(context);
    const unsigned char* bytes = static_cast<const unsigned char*>(data);
    buffer->insert(buffer->end(), bytes, bytes + size);
}*/

std::string MatrixSender::sanitizeUrl(std::string url) {
    while (!url.empty() && url.back() == '/') {
        url.pop_back();
    }
    return url;
}

std::string MatrixSender::readFile(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) return "";
    auto pos = ifs.tellg();
    std::string result(pos, '\0');
    ifs.seekg(0, std::ios::beg);
    ifs.read(&result[0], pos);
    return result;
}

std::string MatrixSender::upload(const std::string& data, const std::string& contentType) const {
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
        SPDLOG_ERROR("Upload failed, status_code: {}, text: {}", r.status_code, r.text);
        return "";
    }

    try {
        auto j = njson::parse(r.text);
        return j["content_uri"];
    } catch (...) {
        return "";
    }
}

void MatrixSender::send_event(const njson& contentBody, const std::string& msgType) const {
    auto now = std::chrono::system_clock::now().time_since_epoch().count();
    const std::string txnId = std::to_string(now);
    const std::string url = homeServer + "/_matrix/client/r0/rooms/" + roomId +
                            "/send/m.room.message/" + txnId;

    njson payload;
    payload["msgtype"] = msgType;
    payload["body"] = contentBody["body"];

    if (contentBody.contains("url"))
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
        SPDLOG_ERROR("Send message failed: {}, {}", r.status_code, r.reason);
    } else {
        SPDLOG_INFO("Send message {} successfully", msgType);
    }
}

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

MatrixSender::MatrixSender(std::string url, std::string token, std::string room) {
    if (url.empty() || token.empty() || room.empty()) {
        throw std::runtime_error("MatrixSender requires URL, Token, and RoomID");
    }
    homeServer = sanitizeUrl(url);
    accessToken = token;
    roomId = room;
}

void MatrixSender::sendText(const std::string& message) {
    if (message.empty()) return;
    njson content;
    content["body"] = message;
    send_event(content, "m.text");
}

void MatrixSender::send_jpeg(const std::string& jpeg_bytes, int width, int height, const std::string& caption) {

    if (std::string mxc = upload(jpeg_bytes, "image/jpeg"); !mxc.empty()) {
        njson content;
        content["body"] = caption;
        content["url"] = mxc;
        content["info"]["w"] = width;
        content["info"]["h"] = height;
        content["info"]["mimetype"] = "image/jpeg";
        content["info"]["size"] = jpeg_bytes.size();
        send_event(content, "m.image");
    }
}

void MatrixSender::send_video(const std::string& filepath, const std::string& caption, int duration_ms) {
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
        send_event(content, "m.video");
    }
}

void MatrixSender::send_video_from_memory(const std::string& video_data,
                                          const std::string& caption,
                                          int duration_ms,
                                          const std::string& thumbnail_data,
                                          int width,
                                          int height,
                                          const std::string& thumb_mime) {
    if (video_data.empty()) return;

    // 1. Upload Video
    std::string video_mxc = upload(video_data, "video/mp4");
    if (video_mxc.empty()) return;

    njson content;
    content["body"] = caption;
    content["url"] = video_mxc;

    njson info;
    info["mimetype"] = "video/mp4";
    info["size"] = video_data.size();
    if (duration_ms > 0) info["duration"] = duration_ms;

    if (width > 0 && height > 0) {
        info["w"] = width;
        info["h"] = height;
    }

    // 2. Upload Thumbnail (if present)
    if (!thumbnail_data.empty()) {
        std::string thumb_mxc = upload(thumbnail_data, thumb_mime);

        if (!thumb_mxc.empty()) {
            info["thumbnail_url"] = thumb_mxc;

            njson thumb_info;
            thumb_info["mimetype"] = thumb_mime;
            thumb_info["size"] = thumbnail_data.size();
            if (width > 0 && height > 0) {
                thumb_info["w"] = width;
                thumb_info["h"] = height;
            }
            info["thumbnail_info"] = thumb_info;
        }
    }

    content["info"] = info;
    send_event(content, "m.video");
}

} // namespace Utils