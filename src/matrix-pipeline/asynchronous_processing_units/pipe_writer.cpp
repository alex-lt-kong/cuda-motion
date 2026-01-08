#include "pipe_writer.h"

#include <spdlog/spdlog.h>

namespace MatrixPipeline::ProcessingUnit {

PipeWriter::~PipeWriter() { close_pipe(); }

bool PipeWriter::init(const njson &config) {
  if (!config.contains("subprocessCmd")) {
    SPDLOG_ERROR("subprocessCmd not defined");
    return false;
  }
  m_subprocess_cmd = config["subprocessCmd"].get<std::string>();
  SPDLOG_INFO("subprocessCmd: {}", m_subprocess_cmd);
  return true;
}

void PipeWriter::on_frame_ready(cv::cuda::GpuMat &gpu_frame,
                                [[maybe_unused]] PipelineContext &ctx) {
  if (gpu_frame.empty()) {
    return;
  }

  // 1. Lazy Initialization of the Pipe
  if (!m_pipe) {
    if (m_subprocess_cmd.empty()) {
      SPDLOG_ERROR("{}: init() was not called with a FFmpeg command.",
                   m_unit_path);
      return;
    }

    SPDLOG_INFO("{}: Opening FFmpeg pipe...", m_unit_path);
    SPDLOG_DEBUG("{}: Command: {}", m_unit_path, m_ffmpeg_cmd);

    // Open pipe to the command
    m_pipe = popen(m_subprocess_cmd.c_str(), "w");

    if (!m_pipe) {
      SPDLOG_ERROR("{}: Failed to open pipe.", m_unit_path);
      return;
    }
  }

  // 2. Download from GPU to CPU
  // fwrite requires a pointer to host memory, so we must download the frame.
  // We reuse m_cpu_frame to avoid allocation overhead on every frame.
  gpu_frame.download(m_cpu_frame);

  // 3. Write raw frame data to the pipe
  // We write exactly (width * height * channels * elemSize) bytes per frame.
  size_t total_bytes = m_cpu_frame.total() * m_cpu_frame.elemSize();
  size_t written = fwrite(m_cpu_frame.data, 1, total_bytes, m_pipe);

  // 4. Health Check
  if (written != total_bytes || ferror(m_pipe)) {
    SPDLOG_ERROR("{}: FFmpeg pipe broken (server disconnected?). Closing pipe.",
                 m_unit_path);
    close_pipe();
    // The pipe will remain null and attempt to reopen on the next frame
  }
}

void PipeWriter::close_pipe() noexcept {
  if (m_pipe) {
    SPDLOG_INFO("pclose()ing pipe.");
    int retries = 10;
    int pclose_ret;
    do {
      errno = 0;
      pclose_ret = pclose(m_pipe);
    } while (pclose_ret == -1 && errno == EINTR && --retries > 0);
    if (pclose_ret == 0)
      SPDLOG_INFO("FFmpeg pipe pclose()ed");
    else
      SPDLOG_ERROR("pclose()ing FFmpeg pipe error: pclose_ret: {}, errno: {} "
                   "({}), but there is nothing else we can do",
                   pclose_ret, errno, strerror(errno));
    m_pipe = nullptr;
  }
}

} // namespace MatrixPipeline::ProcessingUnit