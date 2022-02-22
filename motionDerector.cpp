#include <iostream>
#include <thread>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "classes/deviceManager.h"

using namespace std;
using namespace cv;



int main() {
  cout << "cv::getBuildInformation():\n" <<  getBuildInformation() << "\n";
  deviceManager* myDevice = new deviceManager("/dev/video0");
  thread th(&deviceManager::startMotionDetection, myDevice);
  th.join();
  //cout << "Press enter to quit" << endl;
  //string t;
  //cin >> t;
  return 0;
}