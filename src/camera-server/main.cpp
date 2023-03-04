#include <iostream>
#include <fstream>
#include <signal.h>
#include <thread>
#include <vector>

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include "deviceManager.h"

using namespace std;
using json = nlohmann::json;

volatile sig_atomic_t done = 0;

void signalCallbackHandler(int signum) {
    if (signum == SIGPIPE) {    
        return;
    }

    spdlog::warn("Signal: {} caught, all threads will quit gracefully", signum);
    done = 1;
};

void register_signal_handlers() {
    struct sigaction act;
    act.sa_handler = signalCallbackHandler;
    sigemptyset(&act.sa_mask);
    act.sa_flags = SA_RESETHAND;
    if (sigaction(SIGINT, &act, 0) + sigaction(SIGABRT, &act, 0) +
        sigaction(SIGTERM, &act, 0) + sigaction(SIGPIPE, &act, 0) < 0) {
        throw runtime_error("sigaction() called failed, errno: " + errno);
    }
}

json load_settings() {
    string homeDir(getenv("HOME"));
    string settingsPath = homeDir + "/.config/ak-studio/motion-detector.json";    
    ifstream is(settingsPath);
    json settings;
    is >> settings;
    return settings;
}

int main() {
    spdlog::set_pattern("%Y-%m-%dT%T%z | %t | %8l | %v");
    spdlog::info("Camera Server started");

    register_signal_handlers();
    
    spdlog::info("cv::getBuildInformation(): {}", getBuildInformation());

    json settings = load_settings();

    size_t deviceCount = settings["devices"].size();
    vector<deviceManager> myDevices(deviceCount, deviceManager());
    vector<thread> deviceThreads;

    for (size_t i = 0; i < deviceCount; i++) {
        spdlog::info("Loading {}-th device with the following configs:\n{}",
            i, settings["devices"][i].dump(2));
        myDevices[i].setParameters(settings["devices"][i], &done);
        deviceThreads.emplace_back(
            &deviceManager::startMotionDetection, myDevices[i]);
    }
    for (size_t i = 0; i < settings["devices"].size(); i++) {
        deviceThreads[i].join();
        spdlog::info("{}-th device thread exited gracefully", i);
    }
    spdlog::info("All device threads exited gracefully");


    return 0;  
}
