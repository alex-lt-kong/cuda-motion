#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

class deviceManager {

public:
  deviceManager(string deviceUrl);
  bool captureImage(string imageSaveTo);
  void startMotionDetection();
  void stopMotionDetection();

private:
  string deviceUrl = "";
  bool stopSignal = false;
  
  string convertToString(char* a, int size);
  string CurrentDate();
  void overlayDatetime(Mat frame);
  void overlayChangeRate(Mat frame, float changeRate); 
};