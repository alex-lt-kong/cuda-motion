#include <arpa/inet.h>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <opencv2/core/core.hpp>
#include <regex>
#include <sstream>
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
  this->enableContoursDrawing = settings["frame"]["enableContoursDrawing"];
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


float deviceManager::getFrameChanges(Mat prevFrame, Mat currFrame, Mat* diffFrame) {
  if (prevFrame.empty() || currFrame.empty()) { return -1; }
  if (prevFrame.cols != currFrame.cols || prevFrame.rows != currFrame.rows) { return -1; }
  if (prevFrame.cols == 0 || prevFrame.rows == 0) { return -1; }
  
  absdiff(prevFrame, currFrame, *diffFrame);
  cvtColor(*diffFrame, *diffFrame, COLOR_BGR2GRAY);
  threshold(*diffFrame, *diffFrame, this->pixelLevelThreshold, 255, THRESH_BINARY);
  int nonZeroPixels = countNonZero(*diffFrame);
  return 100.0 * nonZeroPixels / (diffFrame->rows * diffFrame->cols);
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

void deviceManager::overlayContours(Mat dispFrame, Mat diffFrame) {
  if (diffFrame.empty()) return;
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;

  findContours(diffFrame, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE );
  int idx = 0;
  for( ; idx >= 0; idx = hierarchy[idx][0] ) {
    drawContours(dispFrame, contours, idx, Scalar(255, 255, 255), 0.25, 8, hierarchy);
  }
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

void deviceManager::coolDownReachedZero(
  FILE** ffmpegPipe, long long int* videoFrameCount, string* timestampOnVideoStarts
) {
  if (*ffmpegPipe != nullptr) { // No, you cannot pclose() a nullptr
    pclose(*ffmpegPipe); 
    *ffmpegPipe = nullptr; 
    
    if (this->eventOnVideoEnds.length() > 0) {
      this->myLogger.info(this->deviceName, "video recording ends");
      string commandOnVideoEnds = regex_replace(
        this->eventOnVideoEnds, regex("\\{\\{timestamp\\}\\}"), *timestampOnVideoStarts
      );
      system((commandOnVideoEnds + " &").c_str());
      this->myLogger.info(this->deviceName, "onVideoEnds triggered, command [" + commandOnVideoEnds + "] executed");
    } else {
      this->myLogger.info(this->deviceName, "video recording ends, no command to execute");
    }
  }
  *videoFrameCount = 0;
}


void deviceManager::rateOfChangeInRange(
  FILE** ffmpegPipe, int* cooldown, string* timestampOnVideoStarts
) {
  *cooldown = this->framesAfterTrigger;
  if (*ffmpegPipe == nullptr) {
    string command = this->ffmpegCommand;
    *timestampOnVideoStarts = this->getCurrentTimestamp();
    command = regex_replace(
      command, regex("\\{\\{timestamp\\}\\}"), *timestampOnVideoStarts
    );
    *ffmpegPipe = popen((command).c_str(), "w");
    
    if (this->eventOnVideoStarts.length() > 0) {
      this->myLogger.info(this->deviceName, "motion detected, video recording begins");
      string commandOnVideoStarts = regex_replace(
        this->eventOnVideoStarts, regex("\\{\\{timestamp\\}\\}"), *timestampOnVideoStarts
      );
      system((commandOnVideoStarts + " &").c_str());
      this->myLogger.info(this->deviceName, "onVideoStarts triggered, command [" + commandOnVideoStarts + "] executed");
    } else {
      this->myLogger.info(this->deviceName, "motion detected, video recording begins, no command to execute");
    }
  }
}

void deviceManager::startMotionDetection() {

  Mat prevFrame, currFrame, dispFrame, diffFrame;
  bool result = false;
  bool isShowingBlankFrame = false;
  VideoCapture cap;
  float rateOfChange = 0.0;
  string timestampOnVideoStarts = "";

  result = cap.open(this->deviceUri);
  this->myLogger.info(this->deviceName, "cap.open(" + this->deviceUri + "): " + to_string(result));
  long long int totalFrameCount = 0;
  long long int videoFrameCount = 0;
  FILE *ffmpegPipe = nullptr;
  int cooldown = 0;
  
  cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G')); 
  // Thought about moving CAP_PROP_FOURCC to config file. But seems they are good just to be hard-coded?
  if (this->framePreferredWidth > 0) { cap.set(CAP_PROP_FRAME_WIDTH, this->framePreferredWidth); }
  if (this->framePreferredHeight > 0) { cap.set(CAP_PROP_FRAME_HEIGHT, this->framePreferredHeight); }
  if (this->framePreferredFps > 0) { cap.set(CAP_PROP_FPS, this->framePreferredFps); }
  

  while (true) {

    result = cap.grab();
    if (this->skipThisFrame() == true) { continue; }
    if (result) { result = result && cap.retrieve(currFrame); }

    if (result == false || currFrame.empty() || cap.isOpened() == false) {
      stringstream ss;
      ss << "Unable to cap.read() a new frame. currFrame.empty(): " << currFrame.empty() <<
        ", cap.isOpened(): " << cap.isOpened() << ". Sleep for 2 sec than then re-open()...";
      this->myLogger.error(this->deviceName, ss.str());
      isShowingBlankFrame = true;
      this_thread::sleep_for(2000ms); // Don't wait for too long, most of the time the device can be re-open()ed immediately
      cap.open(this->deviceUri);
      if (this->framePreferredWidth >0 && this->framePreferredHeight > 0) {
        currFrame = Mat(this->framePreferredHeight, this->framePreferredWidth, CV_8UC3, Scalar(128, 128, 128));
      }
      else {
        // We cant just do this and skip framePreferredWidth and framePreferredHeight
        // problem will occur when piping frames to ffmpeg: In ffmpeg, we pre-define the frame size, which is mostly
        // framePreferredWidth x framePreferredHeight. If the video device is down and we supply a smaller frame, 
        // ffmpeg will wait until there are enough pixels filling the original resolution to write one frame, 
        // causing screen tearing
        currFrame = Mat(540, 960, CV_8UC3, Scalar(128, 128, 128));
      }
      // 960x540, 1280x760, 1920x1080 all have 16:9 aspect ratio.
    } else {
      if (isShowingBlankFrame == true) {
        this->myLogger.info(this->deviceName, "I am back!");
      }
      isShowingBlankFrame = false;
    }

    
    if (totalFrameCount % this->diffFrameInterval == 0) { rateOfChange = this->getFrameChanges(prevFrame, currFrame, &diffFrame); }

    prevFrame = currFrame.clone();
    dispFrame = currFrame.clone();
    if (this->frameRotation != -1 && isShowingBlankFrame == false) { rotate(dispFrame, dispFrame, this->frameRotation); } 
    if (this->enableContoursDrawing) {
      this->overlayContours(dispFrame, diffFrame); // CPU-intensive! Use with care!
    }
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
      this->rateOfChangeInRange(&ffmpegPipe, &cooldown, &timestampOnVideoStarts);
    }
      
    totalFrameCount ++;
    if (cooldown >= 0) {
      cooldown --;
      if (cooldown > 0) { videoFrameCount ++; }
    }
    if (videoFrameCount >= this->maxFramesPerVideo) { cooldown = 0; }
    if (cooldown == 0) { 
      this->coolDownReachedZero(&ffmpegPipe, &videoFrameCount, &timestampOnVideoStarts);
    }  

    if (ffmpegPipe != nullptr) {       
      fwrite(dispFrame.data, 1, dispFrame.dataend - dispFrame.datastart, ffmpegPipe);      
      if (ferror(ffmpegPipe)) {
        this->myLogger.error(this->deviceName, 
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