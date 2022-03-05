#include "classes/motionDetector.h"

int main() {
  motionDetector* myDetector = new motionDetector();
  myDetector->main();
  delete myDetector;
  return 0;  
}