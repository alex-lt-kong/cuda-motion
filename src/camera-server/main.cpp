#include <iostream>
#include <signal.h>
#include "motionDetector.h"
#include "logger.h"

logger myLogger = logger("/var/log/ak-studio/motion-detector/motionDetector.log", false);
volatile sig_atomic_t done = 0;

void signalCallbackHandler(int signum) {
  if (signum == SIGPIPE) {    
    return;
  }
  myLogger.info("main", "Signal " + std::to_string(signum) + " caught\n");
  done = 1;
};

int main() {  
  myLogger.info("main", "motionDetector started");

  struct sigaction act;
  act.sa_handler = signalCallbackHandler;
  sigemptyset(&act.sa_mask);
  act.sa_flags = SA_RESETHAND;
  sigaction(SIGINT, &act, 0);
  sigaction(SIGABRT, &act, 0);
  sigaction(SIGTERM, &act, 0);
  sigaction(SIGPIPE, &act, 0);

  motionDetector* myDetector = new motionDetector(&done);
  myDetector->main();
  delete myDetector;
  return 0;  
}