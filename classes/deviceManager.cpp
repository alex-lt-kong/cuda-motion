#include <arpa/inet.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <sys/socket.h>
#include <thread>


#include "deviceManager.h"
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

using sysclock_t = std::chrono::system_clock;

string deviceManager::CurrentDate()
{
    std::time_t now = sysclock_t::to_time_t(sysclock_t::now());
    //"19700101_000000"
    char buf[16] = { 0 };
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&now));
    
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

deviceManager::deviceManager(string deviceUrl) {
  this->deviceUrl = deviceUrl;
}

void deviceManager::overlayDatetime(Mat frame) {
  time_t now;
  time(&now);
  char buf[sizeof "2011-10-08 07:07:09"];
  strftime(buf, sizeof buf, "%F %T", localtime(&now));
  putText(frame,buf,Point(5,50), FONT_HERSHEY_DUPLEX, 2 ,cv::Scalar(255,255,255),2,false);
}

void deviceManager::overlayChangeRate(Mat frame, float changeRate) {
  int value = changeRate * 100;
  putText(frame,to_string((float)value / 100) + "%",Point(5,frame.rows-5), FONT_HERSHEY_DUPLEX, 2 ,cv::Scalar(255,255,255),2,false);
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
  rotate(frame, frame, ROTATE_90_CLOCKWISE);
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
  int width = 1920;
  int height = 1080;
  int totalPixels = width * height;
  cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
  int count = 0;
  FILE *output = nullptr;
  int cooldown = 0;
  
  while (true) {
    if (this->stopSignal) {
      cout << "stopSignal received, exited" << endl;
      break;
    }

    result = cap.read(currFrame);
    
    if (result == false || currFrame.empty() || cap.isOpened() == false) {
      cout << "Unable to read() a new frame, "
           << "waiting for 10 sec and then trying to re-open() cv::VideoCapture\n";
    }
    rotate(currFrame, currFrame, cv::ROTATE_90_CLOCKWISE);
    if (prevFrame.empty() == false) {
      absdiff(prevFrame, currFrame, diffFrame);
      cvtColor(diffFrame, grayDiffFrame, COLOR_BGR2GRAY);
      threshold(grayDiffFrame, grayDiffFrame, 32, 255, THRESH_BINARY);
      int nonZeroPixels = countNonZero(grayDiffFrame);
      changeRate = 100.0 * nonZeroPixels / totalPixels;
     
      
      this->overlayDatetime(diffFrame);
      this->overlayChangeRate(diffFrame, changeRate);
      imwrite("/tmp/md/diff" + to_string(count) + " .jpg", diffFrame);
    }
    
    prevFrame = currFrame.clone();
    dispFrame = currFrame.clone();
    this->overlayChangeRate(dispFrame, changeRate);
    this->overlayDatetime(dispFrame);
    
    if (changeRate > 10 && changeRate < 50) {
      
      cooldown = 50;
      if (output == nullptr) {
        output = popen(
          ("/usr/bin/ffmpeg -y -f rawvideo -pixel_format bgr24 -video_size 1080x1920 -framerate 10 -i pipe:0 -vcodec h264 /tmp/" + 
          this->CurrentDate() + ".mp4").c_str(), "w");
      }
    }
    
    count ++;
    if (cooldown >= 0) { cooldown --; }
    if (cooldown == 0) { 
      if (output != nullptr) { pclose(output); output = nullptr; } // No, you cannot pclose() a nullptr
    }
    if (cooldown > 0) {
      for (size_t i = 0; i < dispFrame.dataend - dispFrame.datastart; i++)
        fwrite(&dispFrame.data[i], sizeof(dispFrame.data[i]), 1, output);
    }
    if (count > 500) {
      break;
    }
    cout << "count: " << count << ", cooldown: " << cooldown << ", changeRate: " << changeRate << "%\n";
  }

  cap.release();

}

void deviceManager::stopMotionDetection() {
  this->stopSignal = true;
}