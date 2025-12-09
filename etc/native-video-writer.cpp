#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <signal.h>
#include <stdio.h>

using namespace cv;
using namespace std;

static volatile sig_atomic_t e_flag = 0;

static void signal_handler(int signum) {
  char msg[] = "Signal [  ] caught\n";
  msg[8] = '0' + (char)(signum / 10);
  msg[9] = '0' + (char)(signum % 10);
  (void)write(STDIN_FILENO, msg, strlen(msg));
  e_flag = 1;
}

inline void install_signal_handler() {
  // This design canNOT handle more than 99 signal types
  if (_NSIG > 99) {
    fprintf(stderr, "signal_handler() can't handle more than 99 signals\n");
    abort();
  }
  struct sigaction act;
  // Initialize the signal set to empty, similar to memset(0)
  if (sigemptyset(&act.sa_mask) == -1) {
    perror("sigemptyset()");
    abort();
  }
  act.sa_handler = signal_handler;
  /* SA_RESETHAND means we want our signal_handler() to intercept the signal
  once. If a signal is sent twice, the default signal handler will be used
  again. `man sigaction` describes more possible sa_flags. */
  act.sa_flags = SA_RESETHAND;
  // act.sa_flags = 0;
  if (sigaction(SIGINT, &act, 0) == -1 || sigaction(SIGTERM, &act, 0) == -1) {
    perror("sigaction()");
    abort();
  }
}
int main(int argc, const char *argv[]) {
  if (argc != 3) {
    cerr << "Usage : " << argv[0] << "  <Source URI> <Dest path>" << endl;
    return -1;
  }
  install_signal_handler();
  cout << "getBuildInformation():\n" << getBuildInformation() << endl;
  Mat src;
  // use default camera as video source
  VideoCapture cap;
  cap.open(argv[1], cv::CAP_FFMPEG);
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
  writer.open(argv[2], cv::CAP_FFMPEG, codec, fps, src.size(), isColor);
  // check if we succeeded
  if (!writer.isOpened()) {
    cerr << "Could not open the output video file for write\n";
    return -1;
  }
  //--- GRAB AND WRITE LOOP
  cout << "Writing videofile: " << argv[2] << endl
       << "Press Ctrl+C to exit" << endl;
  while (!e_flag) {
    // check if we succeeded
    if (!cap.read(src)) {
      cerr << "ERROR! blank frame grabbed\n";
      break;
    }
    // encode the frame into the videofile stream
    writer.write(src);
  }

  writer.release();
  cout << "writer.release()ed" << endl;
  // the videofile will be closed and released automatically in VideoWriter
  // destructor
  return 0;
}