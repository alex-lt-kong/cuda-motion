#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include <signal.h>
#include <thread>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "classes/deviceManager.h"


using namespace std;
using namespace cv;
using json = nlohmann::json;

void my_handler(int s){
  printf("Caught signal %d\n",s);
}

int main() {

  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = my_handler;
  sigaction(SIGINT, &sigIntHandler, NULL);
  
  cout << "cv::getBuildInformation():\n" <<  getBuildInformation() << "\n";

  string homeDir(getenv("HOME"));
  string settingsPath = homeDir + "/.config/ak-studio/motion-detector.json";
  cout << "settingsPath: " << settingsPath << endl;
  std::ifstream is(settingsPath);
  json jsonSettings;
  is >> jsonSettings;
  deviceManager *myDevices = new deviceManager[jsonSettings["devices"].size()];
  thread deviceThreads[4];

  for (int i = 0; i < jsonSettings["devices"].size(); i++) {
    cout << "Loading " << i << "-th device: " << jsonSettings["devices"][i] << "\n" << endl;
    myDevices[i].setParameters(
        jsonSettings["devices"][i]["url"],
        jsonSettings["devices"][i]["name"],
        jsonSettings["devices"][i]["resolution"],
        jsonSettings["devices"][i]["rotation"]
      );
    deviceThreads[i] = thread(&deviceManager::startMotionDetection, myDevices[i]);
  }

  for (int i = 0; i < jsonSettings["devices"].size(); i++) {
    deviceThreads[i].join();
  }

  delete[] myDevices;
  return 0;
}