#pragma once

#include <NvInfer.h>

#include <memory>

namespace MatrixPipeline::Utils {

struct CudaDeleter {
  void operator()(void *ptr) const { cudaFree(ptr); }
};

template <typename T>
std::unique_ptr<T, CudaDeleter> make_device_unique(const size_t count) {
  T *ptr = nullptr;
  if (cudaMalloc(reinterpret_cast<void **>(&ptr), count * sizeof(T)) !=
      cudaSuccess) {
    throw std::bad_alloc();
  }
  return std::unique_ptr<T, CudaDeleter>(ptr);
}

class TRTLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override;
};

// Use 'extern' to tell other modules this exists elsewhere
extern TRTLogger g_logger;

} // namespace MatrixPipeline::Utils