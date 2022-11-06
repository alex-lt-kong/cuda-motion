#include <string>
#include <nlohmann/json.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "logger.h"
#include <queue>
#include <sys/time.h>
#include <signal.h>

using namespace std;
using namespace cv;
using json = nlohmann::json;

class deviceManager {

public:
  deviceManager();
  bool captureImage(string imageSaveTo);
  void startMotionDetection();
  void stopMotionDetection();
  bool setParameters(json settings, volatile sig_atomic_t* done);

private:
  bool enableContoursDrawing = false;
  double fontScale = 1;
  double rateOfChangeUpper = 0;
  double rateOfChangeLower = 0;
  double pixelLevelThreshold = 0;
  int snapshotFrameInterval = 1;
  int frameRotation = -1;
  int framePreferredWidth = -1;
  int framePreferredHeight = -1;
  int framePreferredFps = -1;
  int frameFpsUpperCap = 1;
  int framesAfterTrigger = 0;
  long long int maxFramesPerVideo = 1;
  int diffFrameInterval = 1;
  int frameIntervalInMs = 24;
  logger myLogger = logger("/var/log/ak-studio/motionDetector.log", false);;
  string deviceUri = "";
  string deviceName = "";   
  string ffmpegCommand = "";
  string snapshotPath = "";
  string eventOnVideoStarts = "";
  string eventOnVideoEnds = "";
  queue<long long int> frameTimestamps;

  volatile sig_atomic_t* done;
  
  bool skipThisFrame();
  string convertToString(char* a, int size);
  string getCurrentTimestamp();
  void rateOfChangeInRange(FILE** ffmpegPipe, int* cooldown, string* timestampOnVideoStarts);
  void coolDownReachedZero(FILE** ffmpegPipe, uint32_t* videoFrameCount, string* timestampOnVideoStarts);
  void overlayDatetime(Mat frame);
  void overlayDeviceName(Mat frame);
  void overlayContours(Mat dispFrame, Mat diffFrame);
  void overlayChangeRate(Mat frame, float changeRate, int cooldown, long long int videoFrameCount);
  float getFrameChanges(Mat prevFrame, Mat currFrame, Mat* diffFrame);
};
