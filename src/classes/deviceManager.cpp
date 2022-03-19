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
    std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", localtime(&now));
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
  this->deviceUri = settings["uri"];
  this->deviceName = settings["name"];
  this->frameRotation = settings["rotation"];
  this->snapshotPath = settings["snapshotPath"];
  this->fontScale = settings["fontScale"];
  this->externalCommand = settings["externalCommand"];
  this->ffmpegCommand = settings["ffmpegCommand"];
  this->rateOfChangeLower = settings["rateOfChange"]["lowerLimit"];
  this->rateOfChangeUpper = settings["rateOfChange"]["upperLimit"];
  return true;
}

deviceManager::deviceManager() {
}

void deviceManager::overlayDatetime(Mat frame) {
  time_t now;
  time(&now);
  char buf[sizeof "1970-01-01 00:00:00"];
  strftime(buf, sizeof buf, "%F %T", localtime(&now));
  cv::Size textSize = getTextSize(buf, FONT_HERSHEY_DUPLEX, this->fontScale, 8 * this->fontScale, nullptr);
  putText(frame, buf, Point(5, textSize.height * 1.05), FONT_HERSHEY_DUPLEX, this->fontScale, Scalar(0,  0,  0  ), 8 * this->fontScale, LINE_AA, false);
  putText(frame, buf, Point(5, textSize.height * 1.05), FONT_HERSHEY_DUPLEX, this->fontScale, Scalar(255,255,255), 4 * this->fontScale, LINE_AA, false);
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


void deviceManager::overlayDeviceName(Mat frame) {

  cv::Size textSize = getTextSize(this->deviceName, FONT_HERSHEY_DUPLEX, this->fontScale, 8 * this->fontScale, nullptr);
  putText(frame, this->deviceName, Point(frame.cols - textSize.width * 1.05, frame.rows - 5), 
          FONT_HERSHEY_DUPLEX, this->fontScale, Scalar(0,  0,  0  ), 8 * this->fontScale, LINE_AA, false);
  putText(frame, this->deviceName, Point(frame.cols - textSize.width * 1.05, frame.rows - 5),
          FONT_HERSHEY_DUPLEX, this->fontScale, Scalar(255,255,255), 4 * this->fontScale, LINE_AA, false);
}

void deviceManager::overlayChangeRate(Mat frame, float changeRate, int cooldown) {
  int value = changeRate * 100;
  stringstream ssChangeRate;
  ssChangeRate << fixed << setprecision(2) << changeRate;
  putText(frame, ssChangeRate.str() + "% (" + to_string(cooldown) + ")", 
          Point(5, frame.rows-5), FONT_HERSHEY_DUPLEX, this->fontScale, Scalar(0,   0,   0  ), 10, LINE_AA, false);
  putText(frame, ssChangeRate.str() + "% (" + to_string(cooldown) + ")",
          Point(5, frame.rows-5), FONT_HERSHEY_DUPLEX, this->fontScale, Scalar(255, 255, 255), 4, LINE_AA, false);
}

void deviceManager::startMotionDetection() {
  this->stopSignal = false;
  Mat prevFrame, currFrame, diffFrame, grayDiffFrame, dispFrame;
  bool result = false;
  VideoCapture cap;
  float changeRate = 0.0;
  
  result = cap.open(this->deviceUri);
  this->myLogger.info("cap.open(" + this->deviceUri + "): " + to_string(result));
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
      this->myLogger.error(
        "Unable to cap.read() a new frame from device [" + this->deviceName + "]. result: " + 
        to_string(result) + ", currFrame.empty(): " + to_string(currFrame.empty()) +
        ", cap.isOpened(): " + to_string(cap.isOpened()) + ". Sleep for 2 sec than then re-open()...");
      this_thread::sleep_for(2000ms); // Don't wait for too long, most of the time the device can be re-open()ed immediately
      cap.open(this->deviceUri);
      currFrame = Mat(960, 540, CV_8UC3, Scalar(128, 128, 128));
      // 960x540, 1280x760, 1920x1080 all have 16:9 aspect ratio.
    }
    
    if (this->frameRotation != -1) { rotate(dispFrame, dispFrame, ROTATE_90_CLOCKWISE); } 
    
    if (prevFrame.empty() == false &&
        (prevFrame.cols == currFrame.cols && prevFrame.rows == currFrame.rows)) {
      absdiff(prevFrame, currFrame, diffFrame);
      cvtColor(diffFrame, grayDiffFrame, COLOR_BGR2GRAY);
      threshold(grayDiffFrame, grayDiffFrame, 32, 255, THRESH_BINARY);
      int nonZeroPixels = countNonZero(grayDiffFrame);
      changeRate = 100.0 * nonZeroPixels / (grayDiffFrame.rows * grayDiffFrame.cols);
    }

    prevFrame = currFrame.clone();
    dispFrame = currFrame.clone();
    this->overlayChangeRate(dispFrame, changeRate, cooldown);
    this->overlayDatetime(dispFrame);    
    this->overlayDeviceName(dispFrame);

    if (frameCount % 37 == 0) { imwrite(this->snapshotPath, dispFrame); }
    
    if (changeRate > this->rateOfChangeLower && changeRate < this->rateOfChangeUpper) {      
      cooldown = 500;
      if (output == nullptr) {
        string command = this->ffmpegCommand;
        command = regex_replace(command, regex("__timestamp__"), this->getCurrentTimestamp());
        output = popen((command).c_str(), "w");
        if (this->externalCommand.length() > 0) { system((this->externalCommand + " &").c_str()); }
        this->myLogger.info("Device [" + this->deviceName + "] motion detected, video recording begins");
      }
    }
    
    frameCount ++;
    if (cooldown >= 0) { cooldown --; }
    if (cooldown == 0) { 
      if (output != nullptr) { 
        // No, you cannot pclose() a nullptr
          pclose(output); 
          output = nullptr; 
          this->myLogger.info("Device [" + this->deviceName + "] video recording ends");
        } 
    }
      
    if (output != nullptr) { fwrite(dispFrame.data, 1, dispFrame.dataend - dispFrame.datastart, output); }

    stringstream ssChangeRate;
    ssChangeRate << fixed << setprecision(2) << changeRate;
    this->myLogger.debug(
      this->deviceName + ": frameCount: " + to_string(frameCount) + 
      ", cooldown: " + to_string(cooldown) +  ", changeRate: " + ssChangeRate.str());
  }
  cap.release();

}

void deviceManager::stopMotionDetection() {
  this->stopSignal = true;
}

/*
auto start = std::chrono::high_resolution_clock::now();  

auto finish = std::chrono::high_resolution_clock::now();
auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(finish-start);
std::cout << microseconds.count() / 1000 << "ms\n";
*/