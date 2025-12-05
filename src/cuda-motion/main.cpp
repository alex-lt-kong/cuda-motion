#include "device_manager.h"
#include "global_vars.h"
#include "utils.h"

#include <cxxopts.hpp>
#include <drogon/drogon.h>
#include <nlohmann/json.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <spdlog/spdlog.h>

#include <fstream>
#include <iostream>
#include <vector>

using namespace drogon;
using namespace std;
using json = nlohmann::json;

string configPath =
    string(getenv("HOME")) + "/.config/ak-studio/cuda-motion.jsonc";

void signal_handler_cb(__attribute__((unused)) int signum) {
  drogon::app().quit();
  ev_flag = 1;
}

int main(int argc, char *argv[]) {
  cxxopts::Options options(argv[0], "video feed handler that uses CUDA");
  // clang-format off
  options.add_options()
    ("h,help", "print help message")
    ("c,config-path", "path of the config file", cxxopts::value<string>()->default_value(configPath));
  // clang-format on
  auto result = options.parse(argc, argv);
  if (result.count("help") || !result.count("config-path")) {
    std::cout << options.help() << "\n";
    return 0;
  }
  configPath = result["config-path"].as<std::string>();

  // Doc: https://github.com/gabime/spdlog/wiki/3.-Custom-formatting
  // Including microseconds is handy for naive profiling
  spdlog::set_pattern("%Y-%m-%dT%T.%f%z | %8l | %s:%# (%!) | %v");
  SPDLOG_INFO("Cuda Motion started (git commit: {})", GIT_COMMIT_HASH);
  CudaMotion::Utils::install_signal_handler(signal_handler_cb);
  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
  SPDLOG_INFO("cv::getBuildInformation(): {}",
              string(cv::getBuildInformation()));

  SPDLOG_INFO("Loading json settings from {}", configPath);
  ifstream is(configPath);
  settings = json::parse(is, nullptr, true, true);
  auto mgr = make_unique<CudaMotion::DeviceManager>();
  mgr->StartEv();

  // naive way to handle race condition of drogon::app().run(),
  this_thread::sleep_for(chrono::seconds(1));
  app()
      .setLogLevel(trantor::Logger::kWarn)
      .setThreadNum(4)
      .disableSigtermHandling()
      .run();
  SPDLOG_INFO("Drogon exited");

  mgr->JoinEv();

  SPDLOG_INFO("All device event loop threads exited gracefully");

  return 0;
}
