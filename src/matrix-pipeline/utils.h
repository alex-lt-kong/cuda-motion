#pragma once

#include <drogon/drogon.h>
#include <opencv2/cudaimgproc.hpp>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <cpr/cpr.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>


namespace MatrixPipeline::Utils {
using njson = nlohmann::json;
typedef auto(*signal_handler_callback)(int) -> void;

void install_signal_handler(signal_handler_callback cb);

}