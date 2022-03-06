#include <arpa/inet.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <opencv2/core/core.hpp>
#include <regex>
#include <sys/socket.h>
#include <thread>

#include "deviceManager.h"
#include "easylogging++.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using json = nlohmann::json;
using namespace std;
using namespace cv;

using sysclock_t = std::chrono::system_clock;

string deviceManager::getCurrentTimestamp()
{
    std::time_t now = sysclock_t::to_time_t(sysclock_t::now());
    //"19700101_000000"
    char buf[16] = { 0 };
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&now));
    // https://www.cplusplus.com/reference/ctime/strftime/
    return std::string(buf);
}


string deviceManager::convertToString(char* a, int size)
{
    int i;
    string s = "";
    for (i = 0; i < size; i++) {
        s = s + a[i];
    }
    return s;
}

bool deviceManager::setParameters(json settings) {
  this->deviceUrl = settings["url"];
  this->deviceName = settings["name"];
  this->originalFrameHeight = settings["originalResolution"]["height"];
  this->originalFrameWidth = settings["originalResolution"]["width"];
  this->frameRotation = settings["rotation"];
  this->snapshotPath = settings["snapshotPath"];
  this->fontScale = settings["fontScale"];
  this->externalCommand = settings["externalCommand"];
  this->videoDirectory = settings["videoDirectory"];
  this->ffmpegCommand = settings["ffmpegCommand"];
  this->videoExtension = settings["videoExtension"];
  return true;
}

deviceManager::deviceManager() {
}

void deviceManager::overlayDatetime(Mat frame) {
  time_t now;
  time(&now);
  char buf[sizeof "1970-01-01 00:00:00"];
  strftime(buf, sizeof buf, "%F %T", localtime(&now));
  putText(frame, buf, Point(5, 40), FONT_HERSHEY_DUPLEX, this->fontScale, Scalar(0,  0,  0  ), 10, LINE_AA, false);
  putText(frame, buf, Point(5, 40), FONT_HERSHEY_DUPLEX, this->fontScale, Scalar(255,255,255), 4, LINE_AA, false);
  /*
  void cv::putText 	(InputOutputArray  	img,
                    const String &  	text,
                    Point  	org,
                    int  	fontFace,
                    double  	fontScale,
                    Scalar  	color,
                    int  	thickness = 1,
                    int  	lineType = LINE_8,
                    bool  	bottomLeftOrigin = false 
	) 	
  */
}

void deviceManager::overlayChangeRate(Mat frame, float changeRate, int cooldown) {
  int value = changeRate * 100;
  stringstream ssChangeRate;
  ssChangeRate << fixed << setprecision(2) << changeRate;
  putText(frame, ssChangeRate.str() + "% (" + to_string(cooldown) + ")", 
          Point(5,frame.rows-5), FONT_HERSHEY_DUPLEX, this->fontScale, Scalar(0,   0,   0  ), 10, LINE_AA, false);
  putText(frame, ssChangeRate.str() + "% (" + to_string(cooldown) + ")",
          Point(5,frame.rows-5), FONT_HERSHEY_DUPLEX, this->fontScale, Scalar(255, 255, 255), 4, LINE_AA, false);
}

bool deviceManager::captureImage(string imageSaveTo) {
  
  Mat frame;
  bool result = false;
  VideoCapture cap;
  // a fast way to check supported resolution of a camera:
  // v4l2-ctl -d /dev/video0 --list-formats-ext
  
  result = cap.open(this->deviceUrl);
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
  if (result) { cout << "Device [" << this->deviceUrl << "] opened" << endl; }
  else { 
    cout << "Device [" << this->deviceUrl << "] NOT opened, aborted" << endl;
    return false;
  }

  result = cap.read(frame);
  if (result == false || frame.empty()) { 
    cout << "Failed to read() a frame from video source, aborted" << endl;
    return 1;
  }
  if (this->frameRotation != -1) {
    rotate(frame, frame, this->frameRotation);
  }
  this->overlayDatetime(frame);
  

  result = imwrite(imageSaveTo, frame);
  cap.release();

  cout << result << endl;
  return 0;

  return true;
}

void deviceManager::startMotionDetection() {
  this->stopSignal = false;
  Mat prevFrame, currFrame, diffFrame, grayDiffFrame, dispFrame;
  bool result = false;
  VideoCapture cap;
  float changeRate = 0.0;
  // a fast way to check supported resolution of a camera:
  // v4l2-ctl -d /dev/video0 --list-formats-ext
  
  result = cap.open(this->deviceUrl);
  cout << "cap.open(this->deviceUrl): " << result << endl;
  int totalPixels = this->originalFrameHeight * this->originalFrameWidth;
  cap.set(CAP_PROP_FRAME_WIDTH, this->originalFrameWidth);
  cap.set(CAP_PROP_FRAME_HEIGHT, this->originalFrameHeight);
  long long int frameCount = 0;
  FILE *output = nullptr;
  int cooldown = 0;
  
  while (true) {
    if (this->stopSignal) {
      cout << "stopSignal received, exited" << endl;
      break;
    }

    result = cap.read(currFrame);
    
    if (result == false || currFrame.empty() || cap.isOpened() == false) {
      cout << "(result == false || currFrame.empty() || cap.isOpened() == false)" << endl;
      currFrame = Mat(this->originalFrameHeight, this->originalFrameWidth, CV_8UC3, Scalar(128, 128, 128));
    }
    if (this->frameRotation != -1) { rotate(currFrame, currFrame, ROTATE_90_CLOCKWISE); }
    if (prevFrame.empty() == false) {
      absdiff(prevFrame, currFrame, diffFrame);
      cvtColor(diffFrame, grayDiffFrame, COLOR_BGR2GRAY);
      threshold(grayDiffFrame, grayDiffFrame, 32, 255, THRESH_BINARY);
      int nonZeroPixels = countNonZero(grayDiffFrame);
      changeRate = 100.0 * nonZeroPixels / totalPixels;   
      this->overlayDatetime(diffFrame);
      this->overlayChangeRate(diffFrame, changeRate, -1);
    }
    
    prevFrame = currFrame.clone();
    dispFrame = currFrame.clone();
    this->overlayChangeRate(dispFrame, changeRate, cooldown);
    this->overlayDatetime(dispFrame);
    imwrite(this->snapshotPath, dispFrame);
    if (changeRate > 10 && changeRate < 40) {      
      cooldown = 50;
      if (output == nullptr) {
        string command = this->ffmpegCommand;
        command = regex_replace(command, regex("__width__"), to_string(dispFrame.cols));
        command = regex_replace(command, regex("__height__"), to_string(dispFrame.rows));
        command = regex_replace(command, regex("__framerate__"), to_string(this->frameRate));
        command = regex_replace(command, regex("__videoPath__"), this->videoDirectory + "/" + this->deviceName + "_" + this->getCurrentTimestamp());
        command = regex_replace(command, regex("__videoExt__"), this->videoExtension);
        output = popen((command).c_str(), "w");
        if (this->externalCommand.length() > 0) { system(("nohup " + this->externalCommand + " &").c_str()); }
      }
    }
    
    frameCount ++;
    if (cooldown >= 0) { cooldown --; }
    if (cooldown == 0) { 
      if (output != nullptr) { pclose(output); output = nullptr; } // No, you cannot pclose() a nullptr
    }
    if (cooldown > 0) {
      for (size_t i = 0; i < dispFrame.dataend - dispFrame.datastart; i++)
        fwrite(&dispFrame.data[i], sizeof(dispFrame.data[i]), 1, output);
    }
    if (frameCount > 2147483647) {
      break;
    }
    stringstream ssChangeRate;
    ssChangeRate << fixed << setprecision(2) << changeRate;
    LOG(INFO) << "deviceName: "   << setw (10)  << this->deviceName << 
            ", frameCount: " << setw (3)  << frameCount <<
            ", cooldown: "   << setw (3)  << cooldown   <<
            ", changeRate: " << ssChangeRate.str();
  }

  cap.release();

}

void deviceManager::stopMotionDetection() {
  this->stopSignal = true;
}