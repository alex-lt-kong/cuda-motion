#ifndef CM_UTILS_H
#define CM_UTILS_H

#include <spdlog/spdlog.h>

#include <memory>
#include <string>

void execExternalProgramAsync(std::mutex &mtx, const std::string cmd,
                              const std::string &deviceName);

std::string getCurrentTimestamp() noexcept;

void install_signal_handler();

#endif /* CM_UTILS_H */
