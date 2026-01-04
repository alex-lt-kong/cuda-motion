#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
  const std::string rtsp_out = "rtsp://127.0.0.1:8554/mystream";

  // 1. Open Input (Remote RPi Camera)
  cv::VideoCapture cap("http://user:jUDbB4Xs@rpi-door.hk.lan:8554/");
  if (!cap.isOpened()) {
    std::cerr << "Error: Could not open camera." << std::endl;
    return -1;
  }

  int width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  double fps = 30.0;

  // 2. Setup Output (Direct to MediaMTX)
  // We use CAP_FFMPEG backend.
  // We use 'H264' FourCC code.
  cv::VideoWriter writer(rtsp_out, cv::CAP_FFMPEG,
                         cv::VideoWriter::fourcc('H', '2', '6', '4'), fps,
                         cv::Size(width, height), true);

  if (!writer.isOpened()) {
    std::cerr << "Error: Could not open RTSP writer. OpenCV might not support "
                 "RTSP output on this build."
              << std::endl;
    return -1;
  }

  std::cout << "Streaming to " << rtsp_out << " using native OpenCV..."
            << std::endl;

  cv::Mat frame;
  while (true) {
    cap >> frame;
    if (frame.empty())
      break;

    // 3. Write directly
    writer.write(frame);
  }

  cap.release();
  writer.release();
  return 0;
}