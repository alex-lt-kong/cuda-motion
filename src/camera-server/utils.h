#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <array>

#include <spdlog/spdlog.h>

using namespace std;

enum MotionDetectionMode {
  ALWAYS_RECORD = 2,
  DETECT_MOTION = 1,  
  DISABLED = 0
};

typedef void (*exec_cb)(void* This, string stdout, string stderr, int rc); // type for conciseness

string exec(const string& cmd);
void exec_async(void* This, const vector<string>& args, exec_cb cb);
