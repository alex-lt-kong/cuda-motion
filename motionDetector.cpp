#include <iostream>
#include <thread>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "classes/deviceManager.h"

using namespace std;
using namespace cv;



int main() {
  cout << "cv::getBuildInformation():\n" <<  getBuildInformation() << "\n";
  deviceManager* myDevice0 = new deviceManager("/dev/video0", "cor", "1080x1920", ROTATE_90_CLOCKWISE);
  deviceManager* myDevice1 = new deviceManager("/dev/v4l/by-path/pci-0000:00:14.0-usb-0:3.1:1.0-video-index0", "win", "1920x1080", -1);
  thread th0(&deviceManager::startMotionDetection, myDevice0);
  thread th1(&deviceManager::startMotionDetection, myDevice1);
  th0.join();
  th1.join();
  return 0;
}