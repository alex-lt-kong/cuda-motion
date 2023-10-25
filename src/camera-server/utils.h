#ifndef CS_UTILS_H
#define CS_UTILS_H

#include <spdlog/spdlog.h>

#include <memory>
#include <string>

typedef void (*exec_cb)(void *This, std::string stdout, std::string stderr,
                        int rc); // type for conciseness

void execAsync(void *This, const std::vector<std::string> &args, exec_cb cb);

std::string getCurrentTimestamp();

#endif /* CS_UTILS_H */
