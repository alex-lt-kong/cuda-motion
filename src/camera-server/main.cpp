#include <iostream>
#include <signal.h>

#include <spdlog/spdlog.h>

#include "motionDetector.h"


volatile sig_atomic_t done = 0;

void signalCallbackHandler(int signum) {
    if (signum == SIGPIPE) {    
        return;
    }

    spdlog::warn("Signal: {} caught, all threads will quit gracefully", signum);
    done = 1;
};

int main() {
    spdlog::set_pattern("%Y-%m-%dT%T%z | %t | %8l | %v");
    spdlog::info("Camera Server started");

    struct sigaction act;
    act.sa_handler = signalCallbackHandler;
    sigemptyset(&act.sa_mask);
    act.sa_flags = SA_RESETHAND;
    sigaction(SIGINT, &act, 0);
    sigaction(SIGABRT, &act, 0);
    sigaction(SIGTERM, &act, 0);
    sigaction(SIGPIPE, &act, 0);

    motionDetector myDetector = motionDetector(&done);
    myDetector.main();
    return 0;  
}
