#pragma once

#include <string>
#include <vector>
#include <nlohmann/json.hpp>

// Define alias if not already defined in a common header
using njson = nlohmann::json;
using uchar = unsigned char;

namespace MatrixPipeline::Utils {

class MatrixSender {
private:
  std::string homeServer;
  std::string accessToken;
  std::string roomId;

  // Helper to write STB image data (if needed for future GIF support)
  static void stbiWriteFunc(void *context, void *data, int size);

  // Internal helper to perform HTTP POST for media
  std::string upload(const std::string& data, const std::string& contentType) const;

  // Internal helper to perform HTTP PUT for events
  bool send_event(const njson& contentBody, const std::string& msgType) const;

  // Helper to read file from disk
  std::string readFile(const std::string& path);

  // Helper to clean URL strings
  static std::string sanitizeUrl(std::string url);

public:
  // Constructor
  MatrixSender(std::string url, std::string token, std::string room);

  // Send plain text message
  void sendText(const std::string& message);

  // Send JPEG image
  void send_jpeg(const std::string& jpeg_bytes, int width, int height, const std::string& caption = "Image");

  // Send Video from file path
  void send_video(const std::string& filepath, const std::string& caption, int duration_ms = 0);

  // Send Video from memory (Updated to support Thumbnails)
  void send_video_from_memory(const std::string& video_data,
                              const std::string& caption,
                              size_t duration_ms = 0,
                              const std::string& thumbnail_data = {},
                              int width = 0,
                              int height = 0,
                              const std::string& thumb_mime = "image/jpeg") const;
};

} // namespace Utils