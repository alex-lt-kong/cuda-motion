#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

int main(int, char **) {
  cout << "getBuildInformation():\n" << getBuildInformation() << endl;
  Mat src;
  // use default camera as video source
  VideoCapture cap(0);
  // check if we succeeded
  if (!cap.isOpened()) {
    cerr << "ERROR! Unable to open camera\n";
    return -1;
  }
  // get one frame from camera to know frame size and type
  cap >> src;
  // check if we succeeded
  if (src.empty()) {
    cerr << "ERROR! blank frame grabbed\n";
    return -1;
  }
  bool isColor = (src.type() == CV_8UC3);
  cout << "isColor: " << isColor << endl;
  //--- INITIALIZE VIDEOWRITER
  VideoWriter writer;
  int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
  double fps = 25.0; // framerate of the created video stream
  string filename = "/tmp/test.avi";
  writer.open(filename, cv::CAP_FFMPEG, codec, fps, src.size(), isColor);
  // check if we succeeded
  if (!writer.isOpened()) {
    cerr << "Could not open the output video file for write\n";
    return -1;
  }
  //--- GRAB AND WRITE LOOP
  cout << "Writing videofile: " << filename << endl
       << "Press any key to terminate" << endl;
  for (;;) {
    // check if we succeeded
    if (!cap.read(src)) {
      cerr << "ERROR! blank frame grabbed\n";
      break;
    }
    // encode the frame into the videofile stream
    writer.write(src);
  }
  // the videofile will be closed and released automatically in VideoWriter
  // destructor
  return 0;
}
