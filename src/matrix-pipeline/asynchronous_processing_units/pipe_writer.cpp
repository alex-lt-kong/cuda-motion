#include "pipe_writer.h"
#include "../global_vars.h"

#include <spdlog/spdlog.h>

#include <cerrno>

namespace MatrixPipeline::ProcessingUnit {

PipeWriter::~PipeWriter() { close_pipe(); }

bool PipeWriter::init(const njson &config) {
  if (!config.contains("subprocessCmd")) {
    SPDLOG_ERROR("subprocessCmd not defined");
    return false;
  }
  m_subprocess_cmd = config["subprocessCmd"].get<std::string>();
  SPDLOG_INFO("subprocessCmd: {}", m_subprocess_cmd);

  SPDLOG_INFO("{}: Opening FFmpeg pipe...", m_unit_path);
  SPDLOG_DEBUG("popen({})ing...", m_subprocess_cmd);
  m_pipe = popen(m_subprocess_cmd.c_str(), "w");
  if (!m_pipe) {
    SPDLOG_ERROR("{}: Failed to open pipe.", m_unit_path);
    return false;
  }
  SPDLOG_DEBUG("popen({})'ed", m_subprocess_cmd);
  return true;
}

void PipeWriter::on_frame_ready(cv::cuda::GpuMat &gpu_frame,
                                [[maybe_unused]] PipelineContext &ctx) {
  if (gpu_frame.empty() || m_pipe == nullptr || ev_flag != 0) {
    return;
  }
  // fwrite requires a pointer to host memory, so we must download the frame.
  // We reuse m_cpu_frame to avoid allocation overhead on every frame.
  gpu_frame.download(m_cpu_frame);

  // 3. Write raw frame data to the pipe
  // We write exactly (width * height * channels * elemSize) bytes per frame.
  size_t total_bytes = m_cpu_frame.total() * m_cpu_frame.elemSize();
  const size_t written = fwrite(m_cpu_frame.data, 1, total_bytes, m_pipe);

  if (written != total_bytes || ferror(m_pipe)) {
    SPDLOG_ERROR("pipe, disable()ing");
    disable();
  }
}

void PipeWriter::close_pipe() noexcept {
  if (m_pipe) {
    // SPDLOG_INFO("pclose()ing pipe.");
    pclose(m_pipe);
    m_pipe = nullptr;
    // SPDLOG_INFO("pipe pclose()ed");
  }
}

} // namespace MatrixPipeline::ProcessingUnit