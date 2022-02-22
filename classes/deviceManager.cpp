#include <chrono>
#include <ctime>
#include <thread>
#include <iostream>
#include "deviceManager.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

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
    if (prevFrame.empty() == false) {
      absdiff(prevFrame, currFrame, diffFrame);
      cvtColor(diffFrame, grayDiffFrame, COLOR_BGR2GRAY);
      threshold(grayDiffFrame, grayDiffFrame, 32, 255, THRESH_BINARY);
      int nonZeroPixels = countNonZero(grayDiffFrame);
      changeRate = 100.0 * nonZeroPixels / totalPixels;
      cout << "change ratio: " << changeRate << "%\n";
      
      this->overlayDatetime(diffFrame);
      this->overlayChangeRate(diffFrame, changeRate);
      imwrite("/tmp/md/diff" + to_string(count) + " .jpg", diffFrame);
    }
    prevFrame = currFrame.clone();
    dispFrame = currFrame.clone();
    this->overlayChangeRate(dispFrame, changeRate);
    this->overlayDatetime(dispFrame);
    imwrite("/tmp/md/prev" + to_string(count) + " .jpg", dispFrame);
    count ++;
    if (count > 50) {
      break;
    }
    
  }
  cap.release();

}

void deviceManager::stopMotionDetection() {
  this->stopSignal = true;
}