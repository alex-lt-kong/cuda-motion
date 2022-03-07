#include <iostream>
#include "classes/motionDetector.h"
#include "classes/logger.h"

int main() {
  logger myLogger = logger("/var/log/ak-studio/motionDetector.log", false);
  myLogger.info("motionDetector started");
  motionDetector* myDetector = new motionDetector();
  myDetector->main();
  delete myDetector;
  return 0;  
}