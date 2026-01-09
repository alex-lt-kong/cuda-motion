#include "pipe_writer.h"
#include "../global_vars.h"

#include <spdlog/spdlog.h>

#include <cerrno>

namespace MatrixPipeline::ProcessingUnit {

PipeWriter::~PipeWriter() { close_pipe(); }

bool PipeWriter::init(const njson &config) {
  const auto subprocess_cmd_key = "subprocessCmd";
  if (!config.contains(subprocess_cmd_key)) {
    SPDLOG_ERROR("{} not defined", subprocess_cmd_key);
    return false;
  }
  m_subprocess_cmd = config[subprocess_cmd_key].get<std::string>();
  SPDLOG_INFO("popen({})ing...", m_subprocess_cmd);
  m_pipe = popen(m_subprocess_cmd.c_str(), "w");
  if (!m_pipe) {
    SPDLOG_ERROR("popen({}) failed: {}({})", m_subprocess_cmd, errno,
                 strerror(errno));
    return false;
  }
  SPDLOG_INFO("popen({})'ed", m_subprocess_cmd);
  return true;
}

void PipeWriter::on_frame_ready(cv::cuda::GpuMat &gpu_frame,
                                [[maybe_unused]] PipelineContext &ctx) {
  if (gpu_frame.empty() || m_pipe == nullptr || ev_flag != 0) {
    return;
  }

  gpu_frame.download(m_cpu_frame);
  const auto total_bytes = m_cpu_frame.total() * m_cpu_frame.elemSize();
  const auto written = fwrite(m_cpu_frame.data, 1, total_bytes, m_pipe);

  if (written != total_bytes || ferror(m_pipe)) {
    SPDLOG_ERROR("fwrite() error, disable()ing this unit");
    disable();
  }
}

void PipeWriter::close_pipe() noexcept {
  if (m_pipe) {
    pclose(m_pipe);
    m_pipe = nullptr;
  }
}

} // namespace MatrixPipeline::ProcessingUnit