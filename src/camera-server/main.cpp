#include <iostream>
#include <fstream>
#include <signal.h>
#include <thread>
#include <vector>

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <crow.h>

#include "deviceManager.h"

using namespace std;
using njson = nlohmann::json;

volatile sig_atomic_t done = 0;
crow::SimpleApp app;
vector<deviceManager> myDevices;

void signal_handler(int signum) {
    if (signum == SIGPIPE) {    
        return;
    }
    spdlog::warn("Signal: {} caught, all threads will quit gracefully", signum);
    app.stop();
    done = 1;
}

void register_signal_handlers() {
    struct sigaction act;
    act.sa_handler = signal_handler;
    sigemptyset(&act.sa_mask);
    act.sa_flags = SA_RESETHAND;
    if (sigaction(SIGINT, &act, 0) + sigaction(SIGABRT, &act, 0) +
        sigaction(SIGTERM, &act, 0) + sigaction(SIGPIPE, &act, 0) < 0) {
        throw runtime_error("sigaction() called failed, errno: " + errno);
    }
}

njson load_settings() {
    string settingsPath = string(getenv("HOME")) + 
        "/.config/ak-studio/camera-server.json";
    spdlog::info("Loading json settings from {}", settingsPath);
    ifstream is(settingsPath);
    njson settings;
    is >> settings;
    return settings;
}

class CustomLogger : public crow::ILogHandler {
public:
    CustomLogger() {}
    void log(string message, crow::LogLevel level) {
        if (level <= crow::LogLevel::INFO) {
            spdlog::info("CrowCpp: {}", message);
        } else if (level < crow::LogLevel::WARNING) {
            spdlog::warn("CrowCpp: {}", message);
        } else {
            spdlog::error("CrowCpp: {}", message);
        }
    }
};

void start_http_server() {

    CustomLogger logger;
    crow::logger::setHandler(&logger);
    app.loglevel(crow::LogLevel::Warning);

    CROW_ROUTE(app, "/")([](){
        return "HTTP service running";
    });

    CROW_ROUTE(app, "/live_image/")([](const crow::request&, crow::response& res){
        res.set_header("Content-Type", "image/jpg");
        vector<uint8_t> encoded_img;
        myDevices[0].getLiveImage(encoded_img);
        res.end(string((char*)(encoded_img.data()), encoded_img.size()));
    });

    app.bindaddr("127.0.0.1").port(54321).signal_clear().run();

}

int main() {
    app.stop();
    spdlog::set_pattern("%Y-%m-%dT%T%z | %t | %8l | %v");
    spdlog::info("Camera Server started");

    register_signal_handlers();
    
    spdlog::info("cv::getBuildInformation(): {}", getBuildInformation());

    njson settings = load_settings();

    size_t deviceCount = settings["devices"].size();
    myDevices = vector<deviceManager>(deviceCount, deviceManager());
    //vector<thread> deviceThreads;

    for (size_t i = 0; i < deviceCount; i++) {
        spdlog::info("Loading {}-th device with the following configs:\n{}",
            i, settings["devices"][i].dump(2));
        myDevices[i].setParameters(settings["devices"][i], &done);
        myDevices[i].StartInternalThread();
    }

    start_http_server();
    for (size_t i = 0; i < settings["devices"].size(); i++) {
        myDevices[i].WaitForInternalThreadToExit();
        spdlog::info("{}-th device thread exited gracefully", i);
    }
    spdlog::info("All device threads exited gracefully");

    return 0;  
}
