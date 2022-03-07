#include <string>
#include <nlohmann/json.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "logger.h"

using namespace std;
using namespace cv;
using json = nlohmann::json;

class deviceManager {

public:
  deviceManager();
  bool captureImage(string imageSaveTo);
  void startMotionDetection();
  void stopMotionDetection();
  bool setParameters(json settings);

private:
  bool stopSignal = false;
  double fontScale = 1;
  int frameRate = 30;
  int frameRotation = -1;
  int originalFrameWidth = -1;   
  int originalFrameHeight = -1;
  logger myLogger = logger("/var/log/ak-studio/motionDetector.log", false);;
  string deviceUrl = "";
  string deviceName = "";   
  string ffmpegCommand = "";
  string snapshotPath = "";
  string externalCommand = "";
  string videoDirectory = "";
  string videoExtension = "";
  
  string convertToString(char* a, int size);
  string getCurrentTimestamp();
  void overlayDatetime(Mat frame);
  void overlayChangeRate(Mat frame, float changeRate, int cooldown);
};