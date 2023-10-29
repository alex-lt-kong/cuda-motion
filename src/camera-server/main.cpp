#include "device_manager.h"
#include "global_vars.h"
#include "http_service/oatpp_entry.h"
#include "utils.h"

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

void print_usage(string binary_name) {

  cerr << "Usage: " << binary_name << " [OPTION]\n\n";

  cerr << "Options:\n"
       << "  --help,        -h        Display this help and exit\n"
       << "  --config-path, -c        JSON configuration file path, default to "
       << configPath << endl;
}

void parse_arguments(int argc, char **argv) {
  static struct option long_options[] = {
      {"config-path", required_argument, 0, 'c'},
      {"help", optional_argument, 0, 'h'},
      {0, 0, 0, 0}};

  int opt, option_index = 0;

  while ((opt = getopt_long(argc, argv, "c:h", long_options, &option_index)) !=
         -1) {
    switch (opt) {
    case 'c':
      if (optarg != NULL) {
        configPath = string(optarg);
      }
      break;
    default:
      print_usage(argv[0]);
      exit(EXIT_FAILURE);
    }
  }
  if (configPath.empty()) {
    print_usage(argv[0]);
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char *argv[]) {
  parse_arguments(argc, argv);
  // Doc: https://github.com/gabime/spdlog/wiki/3.-Custom-formatting
  spdlog::set_pattern("%Y-%m-%dT%T.%e%z|%5t|%8l| %v");
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
