#ifndef CM_UTILS_H
#define CM_UTILS_H

#include <memory>
#include <string>

namespace CudaMotion::Utils {

typedef auto(*signal_handler_callback)(int) -> void;

void execExternalProgramAsync(std::mutex &mtx, const std::string cmd,
                              const std::string &deviceName);

std::string getCurrentTimestamp() noexcept;

void install_signal_handler(signal_handler_callback cb);
} // namespace CudaMotion::Utils
#endif /* CM_UTILS_H */
