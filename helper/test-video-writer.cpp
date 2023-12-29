#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <signal.h>
#include <stdio.h>
#include <thread>

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
  cuda::GpuMat dFrame;
  Mat hFrame;

  Ptr<cudacodec::VideoReader> dReader =
      cudacodec::createVideoReader(string(argv[1]));
  dReader->set(cv::cudacodec::ColorFormat::BGR);
  Ptr<cudacodec::VideoWriter> dWriter = cudacodec::createVideoWriter(
      string(argv[2]), Size(1280, 720), cudacodec::Codec::H264, 25.0,
      cudacodec::ColorFormat::BGR);
  size_t frameCount = 0;
  while (!e_flag) {
    if (!dReader->nextFrame(dFrame)) {
      cerr << "dReader->nextFrame(dFrame) is False" << endl;
      this_thread::sleep_for(10000ms);
      dReader = cudacodec::createVideoReader(string(argv[1]));
    }
    ++frameCount;

    if (!dFrame.empty()) {
      dWriter->write(dFrame);
    } else {
      cerr << "frameCount: " << frameCount << " is empty" << endl;
    }
    if (!dFrame.empty() && frameCount % 100 == 0) {
      cout << "frameCount: " << frameCount << ", size(): " << dFrame.size()
           << ", channels(): " << dFrame.channels() << endl;
    }
  }
  dWriter->release();
  cout << "dWriter->release()ed" << endl;
  return 0;
}
