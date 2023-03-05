#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <array>

#include <spdlog/spdlog.h>

using namespace std;

typedef void (*exec_cb)(void* This, string output); // type for conciseness

string exec(const string& cmd);
void exec_async(void* This, const string& cmd, exec_cb cb);
