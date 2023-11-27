#include "device_manager.h"
#include "global_vars.h"
#include "http_service/oatpp_entry.h"
#include "utils.h"

#include <boost/program_options.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <fstream>
#include <getopt.h>
#include <iostream>
#include <vector>

using namespace std;
using json = nlohmann::json;
namespace po = boost::program_options;

vector<unique_ptr<DeviceManager>> myDevices;
string configPath =
    string(getenv("HOME")) + "/.config/ak-studio/camera-server.jsonc";

void parse_arguments(int argc, char *argv[]) {

  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ("help,h", "print help message")
    ("config-path,c", po::value<std::string>()->default_value(configPath), "config file path");
  // clang-format on
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (po::error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }

  if (vm.count("help")) {
    cout << desc << endl;
    exit(EXIT_SUCCESS);
  }
  if (vm.count("config-path")) {
    configPath = vm["config-path"].as<std::string>();
  }
  if (configPath.empty()) {
    cout << desc << endl;
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
