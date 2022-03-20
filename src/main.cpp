#include <iostream>
#include <signal.h>
#include "classes/motionDetector.h"
#include "classes/logger.h"

logger myLogger = logger("/var/log/ak-studio/motionDetector.log", false);
void signalCallbackHandler(int signum) {
  myLogger.info("Signal SIGPIPE caught, ffmpeg may have crashed");
};

int main() {  
  myLogger.info("motionDetector started");
  signal(SIGPIPE, signalCallbackHandler);
  motionDetector* myDetector = new motionDetector();
  myDetector->main();
  delete myDetector;
  return 0;  
}