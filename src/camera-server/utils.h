#ifndef UTILS_H
#define UTILS_H

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <string>

#include <spdlog/spdlog.h>

using namespace std;

const string logFormat = "%Y-%m-%dT%T.%e%z|%5t|%8l| %v";

enum MotionDetectionMode {
  MODE_ALWAYS_RECORD = 2,
  MODE_DETECT_MOTION = 1,  
  MODE_DISABLED = 0
};

typedef void (*exec_cb)(void* This, string stdout, string stderr, int rc); // type for conciseness

void execAsync(void* This, const vector<string>& args, exec_cb cb);

#endif
