#include "deviceManager.h"
#include <sys/time.h>

class motionDetector {
  private:
    deviceManager *myDevices;
    int deviceCount = -1;
    volatile sig_atomic_t* done = 0;

  public:
    void main();
    motionDetector(volatile sig_atomic_t* done);
};