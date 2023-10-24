#ifndef CS_UTILS_H
#define CS_UTILS_H

#include <spdlog/spdlog.h>

#include <memory>
#include <string>
#include <thread>

enum MotionDetectionMode {
  MODE_ALWAYS_RECORD = 2,
  MODE_DETECT_MOTION = 1,  
  MODE_DISABLED = 0
};

typedef void (*exec_cb)(void* This, std::string stdout, std::string stderr, int rc); // type for conciseness

void execAsync(void* This, const std::vector<std::string>& args, exec_cb cb);

#endif /* CS_UTILS_H */
