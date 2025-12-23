#include <opencv2/core/cuda.hpp> // Required for GpuMat
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>

#include <fstream>
#include <iostream>

int main() {
  // 1. Configuration
  std::string model_path =
      "/apps/var/matrix-pipeline/models/face_detection_yunet_2023mar.onnx";
  std::string input_image_path = "/apps/tmp/image.jpeg";
  std::string output_image_path = "/apps/tmp/output.jpg";

  float score_threshold = 0.9f;
  float nms_threshold = 0.3f;
  int top_k = 5000;

  // Check if model exists
  std::ifstream file(model_path);
  if (!file.good()) {
    std::cerr << "Error: Model file '" << model_path << "' not found!"
              << std::endl;
    return -1;
  }

  // 2. Initialize the Detector (CUDA Backend)
  cv::Ptr<cv::FaceDetectorYN> detector = cv::FaceDetectorYN::create(
      model_path, "", cv::Size(320, 320), score_threshold, nms_threshold, top_k,
      cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA);

  // 3. Load Image (Host/CPU)
  cv::Mat h_image = cv::imread(input_image_path);
  if (h_image.empty()) {
    std::cerr << "Error: Could not read image '" << input_image_path << "'"
              << std::endl;
    return -1;
  }

  // --- GPU UPLOAD START ---
  // cv::cuda::GpuMat d_image;
  // d_image.upload(h_image); // Move data to GPU
  // --- GPU UPLOAD END ---

  // CRITICAL: Update input size
  detector->setInputSize(h_image.size());

  // 4. Inference
  // We pass the GpuMat (d_image) directly.
  // The 'faces' output will be a CPU Mat because it's a list of coordinates
  // (small data). OpenCV handles the logic of keeping the image processing on
  // GPU and returning results to CPU.
  cv::Mat faces;
  detector->detect(h_image, faces);

  std::cout << "Detection complete. Found " << faces.rows << " faces."
            << std::endl;

  // 5. Visualize (Draw on the CPU image 'h_image')
  for (int i = 0; i < faces.rows; i++) {
    // Bounding Box
    int x = int(faces.at<float>(i, 0));
    int y = int(faces.at<float>(i, 1));
    int w = int(faces.at<float>(i, 2));
    int h = int(faces.at<float>(i, 3));

    // Landmarks
    float right_eye_x = faces.at<float>(i, 4);
    float right_eye_y = faces.at<float>(i, 5);
    float left_eye_x = faces.at<float>(i, 6);
    float left_eye_y = faces.at<float>(i, 7);
    float nose_x = faces.at<float>(i, 8);
    float nose_y = faces.at<float>(i, 9);

    float confidence = faces.at<float>(i, 14);

    std::cout << " - Face " << i << " [Conf: " << confidence << "] at (" << x
              << "," << y << ")" << std::endl;

    // Draw on Host Image
    cv::rectangle(h_image, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);

    cv::circle(h_image, cv::Point2f(right_eye_x, right_eye_y), 3,
               cv::Scalar(0, 0, 255), -1);
    cv::circle(h_image, cv::Point2f(left_eye_x, left_eye_y), 3,
               cv::Scalar(0, 0, 255), -1);
    cv::circle(h_image, cv::Point2f(nose_x, nose_y), 3, cv::Scalar(0, 0, 255),
               -1);
  }

  // 6. Save
  cv::imwrite(output_image_path, h_image);

  return 0;
}