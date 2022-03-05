#include "easylogging++.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include <signal.h>
#include <thread>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "motionDetector.h"

INITIALIZE_EASYLOGGINGPP

using namespace std;
using namespace cv;
using json = nlohmann::json;

  static void my_handler(int s){
      cout << "Caught signal: " << s << endl;
    /*   for (int i = 0; i < me->deviceCount; i++) {
        me->myDevices[i].stopMotionDetection();
      }*/
  }


void motionDetector::main() {
  //counter = 2;
/*   struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = motionDetector::my_handler;
  sigaction(SIGINT, &sigIntHandler, NULL);*/
  
  cout << "cv::getBuildInformation():\n" <<  getBuildInformation() << "\n";

  string homeDir(getenv("HOME"));
  string settingsPath = homeDir + "/.config/ak-studio/motion-detector.json";
  cout << "settingsPath: " << settingsPath << endl;
  std::ifstream is(settingsPath);
  json jsonSettings;
  is >> jsonSettings;
  this->deviceCount = jsonSettings["devices"].size();
  this->myDevices = new deviceManager[deviceCount];
  thread deviceThreads[4];

  for (int i = 0; i < deviceCount; i++) {
    cout << "Loading " << i << "-th device: " << jsonSettings["devices"][i] << "\n" << endl;
    myDevices[i].setParameters(
        jsonSettings["devices"][i]["url"],
        jsonSettings["devices"][i]["name"],
        jsonSettings["devices"][i]["resolution"],
        jsonSettings["devices"][i]["rotation"],
        jsonSettings["devices"][i]["snapshotPath"],
        jsonSettings["devices"][i]["fontScale"],
        jsonSettings["devices"][i]["externalCommand"],
        jsonSettings["devices"][i]["videoDirectory"]
      );
    deviceThreads[i] = thread(&deviceManager::startMotionDetection, myDevices[i]);
  }

  for (int i = 0; i < jsonSettings["devices"].size(); i++) {
    deviceThreads[i].join();
  }

  delete[] myDevices;
}
