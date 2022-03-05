#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

class deviceManager {

public:
  deviceManager(string deviceUrl, string deviceName, string frameResolution, int frameRotation);
  deviceManager();
  bool captureImage(string imageSaveTo);
  void startMotionDetection();
  void stopMotionDetection();
  bool setParameters(string deviceUrl, string deviceName, string frameResolution, int frameRotation);

private:
  bool stopSignal = false;
  double fontSize = 1.2;
  int frameRate = 5;
  int frameRotation = -1;
  string deviceUrl = "";
  string deviceName = "";
  string frameResolution = "";   

  
  string convertToString(char* a, int size);
  string CurrentDate();
  void overlayDatetime(Mat frame);
  void overlayChangeRate(Mat frame, float changeRate, int cooldown);
};