#pragma once

#include <string>
#include <cstddef> // for size_t

namespace CudaMotion::Utils {


struct RamVideoBuffer {
  int fd;
  std::string m_virtual_path;
  void* m_data_ptr = nullptr;
  size_t size = 0;

  // Constructor: Creates the memfd and sets m_virtual_path
  RamVideoBuffer();

  // Destructor: Cleans up memory mapping and closes file descriptor
  ~RamVideoBuffer();

  // Locks the file size and memory maps it for reading
  [[nodiscard]]bool lock_and_map();
};

}

