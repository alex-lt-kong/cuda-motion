#ifndef CS_UTILS_H
#define CS_UTILS_H

#include <spdlog/spdlog.h>

#include <memory>
#include <string>

void execExternalProgramAsync(std::mutex &mtx, const std::string &cmd,
                              const std::string &deviceName);

std::string getCurrentTimestamp();

void install_signal_handler();

#endif /* CS_UTILS_H */
