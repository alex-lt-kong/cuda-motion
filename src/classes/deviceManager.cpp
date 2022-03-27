#include <arpa/inet.h>
#include <chrono>
#include <ctime>
#include <filesystem>
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
  this->frameRotation = settings["frame"]["rotation"];
  this->framePreferredWidth = settings["frame"]["preferredWidth"];
  this->framePreferredHeight = settings["frame"]["preferredHeight"];
  this->framePreferredFps = settings["frame"]["preferredFps"];
  this->frameFpsUpperCap = settings["frame"]["FpsUpperCap"];
  this->fontScale = settings["frame"]["overlayTextFontScale"];
  this->snapshotPath = settings["snapshot"]["path"];
  this->snapshotFrameInterval = settings["snapshot"]["frameInterval"];  
  this->eventOnVideoStarts = settings["events"]["onVideoStarts"];
  this->eventOnVideoEnds = settings["events"]["onVideoEnds"];
  this->ffmpegCommand = settings["ffmpegCommand"];
  this->rateOfChangeLower = settings["motionDetection"]["frameLevelRateOfChangeLowerLimit"];
  this->rateOfChangeUpper = settings["motionDetection"]["frameLevelRateOfChangeUpperLimit"];
  this->pixelLevelThreshold = settings["motionDetection"]["pixelLevelDiffThreshold"];
  this->diffFrameInterval = settings["motionDetection"]["diffFrameInterval"];
  this->framesAfterTrigger = settings["video"]["framesAfterTrigger"];
  this->maxFramesPerVideo = settings["video"]["maxFramesPerVideo"];

  this->frameIntervalInMs = 1000 * (1.0 / this->frameFpsUpperCap);
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
  putText(frame, buf, Point(5, textSize.height * 1.05), FONT_HERSHEY_DUPLEX, this->fontScale, Scalar(255,255,255), 2 * this->fontScale, LINE_AA, false);
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


float deviceManager::getRateOfChange(Mat prevFrame, Mat currFrame) {
  if (prevFrame.empty()) { return -1; }
  if (prevFrame.cols != currFrame.cols || prevFrame.rows != currFrame.rows) { return -1; }
  if (prevFrame.cols == 0 || prevFrame.rows == 0) { return -1; }

  Mat diffFrame, grayDiffFrame;
  absdiff(prevFrame, currFrame, diffFrame);
  cvtColor(diffFrame, grayDiffFrame, COLOR_BGR2GRAY);
  threshold(grayDiffFrame, grayDiffFrame, this->pixelLevelThreshold, 255, THRESH_BINARY);
  int nonZeroPixels = countNonZero(grayDiffFrame);
  return 100.0 * nonZeroPixels / (grayDiffFrame.rows * grayDiffFrame.cols);
}

void deviceManager::overlayDeviceName(Mat frame) {

  cv::Size textSize = getTextSize(this->deviceName, FONT_HERSHEY_DUPLEX, this->fontScale, 8 * this->fontScale, nullptr);
  putText(frame, this->deviceName, Point(frame.cols - textSize.width * 1.05, frame.rows - 5), 
          FONT_HERSHEY_DUPLEX, this->fontScale, Scalar(0,  0,  0  ), 8 * this->fontScale, LINE_AA, false);
  putText(frame, this->deviceName, Point(frame.cols - textSize.width * 1.05, frame.rows - 5),
          FONT_HERSHEY_DUPLEX, this->fontScale, Scalar(255,255,255), 2 * this->fontScale, LINE_AA, false);
}

void deviceManager::overlayChangeRate(Mat frame, float changeRate, int cooldown, long long int videoFrameCount) {
  int value = changeRate * 100;
  stringstream ssChangeRate;
  ssChangeRate << fixed << setprecision(2) << changeRate;
  putText(frame, ssChangeRate.str() + "% (" + to_string(cooldown) + ", " + to_string(this->maxFramesPerVideo - videoFrameCount) + ")", 
          Point(5, frame.rows-5), FONT_HERSHEY_DUPLEX, this->fontScale, Scalar(0,   0,   0  ), 8 * this->fontScale, LINE_AA, false);
  putText(frame, ssChangeRate.str() + "% (" + to_string(cooldown) + ", " + to_string(this->maxFramesPerVideo - videoFrameCount) + ")",
          Point(5, frame.rows-5), FONT_HERSHEY_DUPLEX, this->fontScale, Scalar(255, 255, 255), 2 * this->fontScale, LINE_AA, false);
}

bool deviceManager::skipThisFrame() {
  int sampleMsLowerLimit = 1000;
  int sampleMsUpperLimit = 60 * 1000;
  int currMsSinceEpoch = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
  if (this->frameTimestamps.size() <= 1) { 
    this->frameTimestamps.push(currMsSinceEpoch); 
    return false;
  }
  
  float fps = 1000.0 * this->frameTimestamps.size() / (1 + currMsSinceEpoch - this->frameTimestamps.front());
  if (currMsSinceEpoch - this->frameTimestamps.front() > sampleMsUpperLimit) {
    this->frameTimestamps.pop();
  }
  if (fps > this->frameFpsUpperCap) { return true; }
  this->frameTimestamps.push(currMsSinceEpoch);  
  return false;
}

void deviceManager::startMotionDetection() {

  Mat prevFrame, currFrame, dispFrame;
  bool result = false;
  bool isShowingBlankFrame = false;
  VideoCapture cap;
  float rateOfChange = 0.0;

  result = cap.open(this->deviceUri);
  this->myLogger.info("cap.open(" + this->deviceUri + "): " + to_string(result));
  long long int totalFrameCount = 0;
  long long int videoFrameCount = 0;
  FILE *ffmpegPipe = nullptr;
  int cooldown = 0;
  
  cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
  if (this->framePreferredWidth > 0) { cap.set(CAP_PROP_FRAME_WIDTH, this->framePreferredWidth); }
  if (this->framePreferredHeight > 0) { cap.set(CAP_PROP_FRAME_HEIGHT, this->framePreferredHeight); }
  if (this->framePreferredFps > 0) { cap.set(CAP_PROP_FPS, this->framePreferredFps); }
  

  while (true) {

    result = cap.grab();
    if (this->skipThisFrame() == true) { continue; }
    if (result) { result = result && cap.retrieve(currFrame); }

    if (result == false || currFrame.empty() || cap.isOpened() == false) {
      this->myLogger.error(
        "Unable to cap.read() a new frame from device [" + this->deviceName + "]. result: " + 
        to_string(result) + ", currFrame.empty(): " + to_string(currFrame.empty()) +
        ", cap.isOpened(): " + to_string(cap.isOpened()) + ". Sleep for 2 sec than then re-open()...");
      isShowingBlankFrame = true;
      this_thread::sleep_for(2000ms); // Don't wait for too long, most of the time the device can be re-open()ed immediately
      cap.open(this->deviceUri);
      currFrame = Mat(540, 960, CV_8UC3, Scalar(128, 128, 128));
      // 960x540, 1280x760, 1920x1080 all have 16:9 aspect ratio.
    } else {
      if (isShowingBlankFrame == true) {
        this->myLogger.info(" [" + this->deviceName + "] is back!");
      }
      isShowingBlankFrame = false;
    }

    if (totalFrameCount % this->diffFrameInterval == 0) { rateOfChange = this->getRateOfChange(prevFrame, currFrame); }

    prevFrame = currFrame.clone();
    dispFrame = currFrame.clone();
    if (this->frameRotation != -1 && isShowingBlankFrame == false) { rotate(dispFrame, dispFrame, this->frameRotation); } 
    this->overlayChangeRate(dispFrame, rateOfChange, cooldown, videoFrameCount);
    this->overlayDatetime(dispFrame);    
    this->overlayDeviceName(dispFrame);
    
    if (totalFrameCount % this->snapshotFrameInterval == 0) {
      // https://stackoverflow.com/questions/7054844/is-rename-atomic
      // https://stackoverflow.com/questions/29261648/atomic-writing-to-file-on-linux
      string ext = this->snapshotPath.substr(this->snapshotPath.find_last_of(".") + 1);
      imwrite(this->snapshotPath + "." + ext, dispFrame); 
      rename((this->snapshotPath + "." + ext).c_str(), this->snapshotPath.c_str());
    }
    
    if (rateOfChange > this->rateOfChangeLower && rateOfChange < this->rateOfChangeUpper) {      
      cooldown = this->framesAfterTrigger;
      if (ffmpegPipe == nullptr) {
        string command = this->ffmpegCommand;
        command = regex_replace(command, regex("\\{\\{timestamp\\}\\}"), this->getCurrentTimestamp());
        ffmpegPipe = popen((command).c_str(), "w");
        this->myLogger.info("[" + this->deviceName + "] motion detected, video recording begins");
        if (this->eventOnVideoStarts.length() > 0) {
          system((this->eventOnVideoStarts + " &").c_str());
          this->myLogger.info("[" + this->deviceName + "] onVideoStarts triggered, command [" + this->eventOnVideoStarts + "] executed");
        }
        
      }      
    }    
      
    totalFrameCount ++;
    if (cooldown >= 0) {
      cooldown --;
      if (cooldown > 0) { videoFrameCount ++; }
    }
    if (videoFrameCount >= this->maxFramesPerVideo) { cooldown = 0; }
    if (cooldown == 0) { 
      if (ffmpegPipe != nullptr) { // No, you cannot pclose() a nullptr
          pclose(ffmpegPipe); 
          ffmpegPipe = nullptr; 
          this->myLogger.info("[" + this->deviceName + "] video recording ends");
          if (this->eventOnVideoEnds.length() > 0) {            
            system((this->eventOnVideoEnds + " &").c_str());
            this->myLogger.info("[" + this->deviceName + "] onVideoEnds triggered, command [" + this->eventOnVideoEnds + "] executed");
          }
        }
        videoFrameCount = 0;
    }  

    if (ffmpegPipe != nullptr) {       
      fwrite(dispFrame.data, 1, dispFrame.dataend - dispFrame.datastart, ffmpegPipe);      
      if (ferror(ffmpegPipe)) {
        this->myLogger.info(
          "ferror(ffmpegPipe) is true, unable to fwrite() more frames to the pipe (cooldown: "
          + to_string(cooldown) + ")"
        );
      }
    }

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