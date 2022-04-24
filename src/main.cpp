#include <iostream>
#include <signal.h>
#include "classes/motionDetector.h"
#include "classes/logger.h"

logger myLogger = logger("/var/log/ak-studio/motion-detector/motionDetector.log", false);
void signalCallbackHandler(int signum) {
  myLogger.info("main", "Signal SIGPIPE caught, ffmpeg may have crashed");
};

int main() {  
  myLogger.info("main", "motionDetector started");
  signal(SIGPIPE, signalCallbackHandler);
  motionDetector* myDetector = new motionDetector();
  myDetector->main();
  delete myDetector;
  return 0;  
}