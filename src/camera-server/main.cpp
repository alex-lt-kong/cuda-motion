#include "device_manager.h"
#include "global_vars.h"
#include "http_service/oatpp_entry.h"
#include "utils.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

using namespace std;
using json = nlohmann::json;

std::vector<deviceManager *> myDevices;

static void signal_handler(int signum) {
  if (signum != SIGCHLD) {
    ev_flag = 1;
  }
  char msg[] = "Signal [  ] caught\n";
  msg[8] = '0' + signum / 10;
  msg[9] = '0' + signum % 10;
  size_t len = sizeof(msg) - 1;
  size_t written = 0;
  while (written < len) {
    ssize_t ret = write(STDOUT_FILENO, msg + written, len - written);
    if (ret == -1) {
      perror("write()");
      break;
    }
    written += ret;
  }
}

void install_signal_handler() {
  static_assert(_NSIG < 99,
                "signal_handler() can't handle more than 99 signals");

  struct sigaction act;
  // Initialize the signal set to empty, similar to memset(0)
  if (sigemptyset(&act.sa_mask) == -1) {
    perror("sigemptyset()");
    abort();
  }
  act.sa_handler = signal_handler;
  /* SA_RESETHAND means we want our signal_handler() to intercept the signal
  once. If a signal is sent twice, the default signal handler will be used
  again. `man sigaction` describes more possible sa_flags. */
  /* In this particular case, we should not enable SA_RESETHAND, mainly
  due to the issue that if a child process is kill, multiple SIGPIPE will
  be invoked consecutively, breaking the program.  */
  // act.sa_flags = SA_RESETHAND;
  if (sigaction(SIGINT, &act, 0) + sigaction(SIGABRT, &act, 0) +
          sigaction(SIGQUIT, &act, 0) + sigaction(SIGTERM, &act, 0) +
          sigaction(SIGPIPE, &act, 0) + sigaction(SIGCHLD, &act, 0) +
          sigaction(SIGTRAP, &act, 0) <
      0) {
    throw runtime_error("sigaction() called failed: " + to_string(errno) + "(" +
                        strerror(errno) + ")");
  }
}

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
  settings = json::parse(is,
                         /* callback */ nullptr,
                         /* allow exceptions */ true,
                         /* ignore_comments */ true);

  size_t deviceCount = settings["devices"].size();
  if (deviceCount == 0) {
    throw logic_error("No devices are defined.");
  }
  myDevices = vector<deviceManager *>(deviceCount);
  // myDevices.reserve(deviceCount);
  for (size_t i = 0; i < deviceCount; ++i) {
    /* Seems ZeroMQ objects are neither copyable nor movable, using
    pointer is a relatively easy way to circumvent these operations. */
    myDevices[i] = new deviceManager(i, settings["devicesDefault"],
                                     settings["devices"][i]);
    myDevices[i]->StartEv();
  }
  initialize_http_service(settings["httpService"]["interface"].get<string>(),
                          settings["httpService"]["port"].get<int>());
  // start_http_server();
  for (size_t i = 0; i < myDevices.size(); ++i) {
    myDevices[i]->JoinEv();
    spdlog::info("{}-th device thread exited gracefully", i);
    delete myDevices[i];
  }
  stop_http_service();
  spdlog::info("All device threads exited gracefully");

  return 0;
}
