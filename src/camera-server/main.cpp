#include "device_manager.h"
#include "global_vars.h"
#include "http_service/oatpp_entry.h"
#include "utils.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <fstream>
#include <iostream>
#include <vector>

using namespace std;
using json = nlohmann::json;

vector<unique_ptr<DeviceManager>> myDevices;

int main(int argc, char *argv[]) {
  string settingsPath;
  if (argc > 2) {
    cerr << "Usage: " << argv[0] << " [config-file.jsonc]" << endl;
    return EXIT_FAILURE;
  } else if (argc == 2) {
    settingsPath = string(argv[1]);
  } else {
    settingsPath =
        string(getenv("HOME")) + "/.config/ak-studio/camera-server.jsonc";
  }
  // Doc: https://github.com/gabime/spdlog/wiki/3.-Custom-formatting
  spdlog::set_pattern("%Y-%m-%dT%T.%e%z|%5t|%8l| %v");
  spdlog::info("Camera Server started");
  install_signal_handler();

  spdlog::info("cv::getBuildInformation(): {}", string(getBuildInformation()));

  spdlog::info("Loading json settings from {}", settingsPath);
  ifstream is(settingsPath);
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
