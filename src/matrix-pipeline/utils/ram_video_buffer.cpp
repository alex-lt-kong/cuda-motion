#ifndef _GNU_SOURCE
// This must be the very first line to enable memfd_create
#define _GNU_SOURCE
#endif

#include "ram_video_buffer.h"

#include <spdlog/spdlog.h>

#include <fcntl.h> // File control definitions
#include <filesystem>
#include <stdexcept>  // std::runtime_error
#include <sys/mman.h> // mmap, munmap, memfd_create
#include <sys/stat.h> // fstat
#include <unistd.h>   // close

MatrixPipeline::Utils::RamVideoBuffer::RamVideoBuffer() {
  // 1. Create an anonymous file in RAM
  // "mp4_buffer" is just a name for debugging tools, not a real filename
  fd = memfd_create("mp4_buffer", 0);
  if (fd == -1) {
    throw std::runtime_error("memfd_create failed");
  }

  // 2. Generate the magic path that OpenCV can open
  // /proc/self/fd/X is a symlink to the actual open file description
  std::filesystem::path proc_path = "/proc/self/fd";
  m_virtual_path = (proc_path / std::to_string(fd)).string();
}

MatrixPipeline::Utils::RamVideoBuffer::~RamVideoBuffer() {
  // Unmap memory if it was mapped
  if (m_data_ptr && m_data_ptr != MAP_FAILED) {
    if (auto result = munmap(m_data_ptr, size); result != 0) {
      SPDLOG_ERROR("munmap failed: {} (erron: {})", result, errno);
    }
  }

  // Close the file descriptor (this frees the RAM)
  if (fd != -1) {
    close(fd);
  }
}

[[nodiscard]] bool MatrixPipeline::Utils::RamVideoBuffer::lock_and_map() {
  // Get the final size of the video written by OpenCV
  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    SPDLOG_ERROR("fstat failed");
    return false;
  }
  size = sb.st_size;

  // Map it to a pointer
  m_data_ptr = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (m_data_ptr == MAP_FAILED) {
    SPDLOG_ERROR("mmap failed");
    return false;
  }
  return true;
}