#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include <signal.h>
#include <thread>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "motionDetector.h"

using namespace std;
using namespace cv;
using json = nlohmann::json;


void motionDetector::main() {

  logger myLogger = logger("/var/log/ak-studio/motionDetector.log", false);

  myLogger.info("cv::getBuildInformation():\n" + getBuildInformation());

  string homeDir(getenv("HOME"));
  string settingsPath = homeDir + "/.config/ak-studio/motion-detector.json";
  
  std::ifstream is(settingsPath);
  json jsonSettings;
  is >> jsonSettings;
  this->deviceCount = jsonSettings["devices"].size();
  this->myDevices = new deviceManager[deviceCount];
  thread deviceThreads[this->deviceCount];

  for (int i = 0; i < deviceCount; i++) {
    myLogger.info("Loading " + to_string(i) + "-th device: " + jsonSettings["devices"][i].dump(2));
    myDevices[i].setParameters(jsonSettings["devices"][i]);
    deviceThreads[i] = thread(&deviceManager::startMotionDetection, myDevices[i]);
  }

  for (int i = 0; i < jsonSettings["devices"].size(); i++) {
    deviceThreads[i].join();
  }

  delete[] myDevices;
}
