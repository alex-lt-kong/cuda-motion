#include <chrono>
#include <iostream>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <path_to_yunet.onnx> [video_source]\n";
    return -1;
  }

  std::string source = (argc > 2) ? argv[2] : "0";
  cv::VideoCapture cap;
  if (source.length() == 1 && isdigit(source[0])) {
    cap.open(std::stoi(source));
  } else {
    cap.open(source);
  }

  if (!cap.isOpened()) {
    std::cerr << "Failed to open video source.\n";
    return -1;
  }

  // Initialize YuNet directly with the CUDA backend
  cv::Ptr<cv::FaceDetectorYN> detector = cv::FaceDetectorYN::create(
      argv[1], "", cv::Size(320, 320), 0.6f, 0.3f, 5000,
      cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA);

  if (!detector) {
    std::cerr << "Failed to create FaceDetectorYN instance.\n";
    return -1;
  }

  cv::Mat frame;
  cv::Mat faces;
  uint64_t frame_count = 0;

  std::cout << "Starting minimal stress test. Monitor RAM with 'top'.\n";

  auto start_time = std::chrono::steady_clock::now();

  while (true) {
    cap >> frame;
    if (frame.empty()) {
      cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Loop video
      continue;
    }

    // Dynamically update size to match the frame
    detector->setInputSize(frame.size());
    // Execute inference
    detector->detect(frame, faces);

    // Simulate extraction to trigger the allocator growth
    if (!faces.empty()) {
      std::vector<cv::Mat> extracted_faces;
      for (int i = 0; i < faces.rows; ++i) {
        extracted_faces.push_back(faces.row(i).clone());
      }
    }

    if (++frame_count % 500 == 0) {
      auto now = std::chrono::steady_clock::now();
      double elapsed_sec =
          std::chrono::duration<double>(now - start_time).count();
      double fps = 500.0 / elapsed_sec;

      std::cout << "Processed " << frame_count << " frames. FPS: " << fps
                << "\n";

      // Reset timer for the next 500 frames
      start_time = std::chrono::steady_clock::now();
    }
  }

  return 0;
}