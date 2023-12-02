#include "device_manager.h"
#include "global_vars.h"
#include "http_service/oatpp_entry.h"
#include "utils.h"

#include <cxxopts.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <fstream>
#include <getopt.h>
#include <iostream>
#include <vector>

using namespace std;
using json = nlohmann::json;

vector<unique_ptr<DeviceManager>> myDevices;
string configPath =
    string(getenv("HOME")) + "/.config/ak-studio/camera-server.jsonc";

int main(int argc, char *argv[]) {
  cxxopts::Options options(argv[0], "video feed handler that uses CUDA");
  // clang-format off
  options.add_options()
    ("h,help", "print help message")
    ("c,config-path", "path of the config file", cxxopts::value<string>()->default_value(configPath));
  // clang-format on
  auto result = options.parse(argc, argv);
  if (result.count("help") || !result.count("config-file")) {
    std::cout << options.help() << "\n";
    return 0;
  }
  configPath = result["config-path"].as<std::string>();

  // Doc: https://github.com/gabime/spdlog/wiki/3.-Custom-formatting
  // Including microseconds is handy for naive profiling
  spdlog::set_pattern("%Y-%m-%dT%T.%f%z|%5t|%8l| %v");
  spdlog::info("Camera Server started");
  install_signal_handler();

  spdlog::info("cv::getBuildInformation(): {}", string(getBuildInformation()));

  spdlog::info("Loading json settings from {}", configPath);
  ifstream is(configPath);
  settings = json::parse(is, nullptr, true, true);
  size_t deviceCount = settings["devices"].size();
  if (deviceCount == 0) {
    throw logic_error("No devices are defined.");
  }
  myDevices = vector<unique_ptr<DeviceManager>>();
  for (size_t i = 0; i < deviceCount; ++i) {
    myDevices.emplace_back(make_unique<DeviceManager>(i));
    myDevices[i]->StartEv();
  }
  initialize_http_service(settings["httpService"]["interface"].get<string>(),
                          settings["httpService"]["port"].get<int>());

  for (size_t i = 0; i < myDevices.size(); ++i) {
    myDevices[i]->JoinEv();
    spdlog::info("{}-th device event loop thread exited gracefully", i);
  }
  stop_http_service();
  spdlog::info("All device event loop threads exited gracefully");

  return 0;
}
