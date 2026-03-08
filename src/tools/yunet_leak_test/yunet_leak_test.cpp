#include <opencv2/core/cuda.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

// ============================================================================
// 1. MOCKS (To simulate your MatrixPipeline environment without dependencies)
// ============================================================================

#define SPDLOG_INFO(...)                                                       \
  fprintf(stdout, "[INFO] ");                                                  \
  fprintf(stdout, __VA_ARGS__);                                                \
  fprintf(stdout, "\n")
#define SPDLOG_ERROR(...)                                                      \
  fprintf(stderr, "[ERROR] ");                                                 \
  fprintf(stderr, __VA_ARGS__);                                                \
  fprintf(stderr, "\n")

struct njson {
  std::string modelPath;
  float value(const std::string &key, float default_val) const {
    return default_val;
  }
  int value(const std::string &key, int default_val) const {
    return default_val;
  }
  std::string value(const std::string &key, const char *default_val) const {
    if (key == "modelPath")
      return modelPath;
    return default_val;
  }
};

enum SynchronousProcessingResult { success_and_continue, failure_and_stop };

struct YuNetDetection {
  cv::Mat yunet_output;
  cv::Rect2f bounding_box;
  std::pair<float, float> landmarks[5];
  float face_score;
};

struct YuNetSFaceResult {
  YuNetDetection detection;
};

struct PipelineContext {
  struct {
    std::vector<YuNetSFaceResult> results;
    cv::Size yunet_input_frame_size;
  } yunet_sface;
};

// ============================================================================
// 2. YUNET DETECT CLASS (With both optimizations applied)
// ============================================================================

namespace MatrixPipeline::ProcessingUnit {

class YuNetDetect {
public:
  bool init(const njson &config);
  SynchronousProcessingResult process(cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx);

private:
  cv::Ptr<cv::FaceDetectorYN> m_detector;
  float m_face_score_threshold = 0.6f;
  float m_nms_threshold = 0.3f;
  int m_top_k = 5000;

  cv::cuda::HostMem m_pinned_buffer;
  cv::cuda::GpuMat
      m_resized_frame_gpu; // Prevents per-frame cudaMalloc thrashing
};

bool YuNetDetect::init(const njson &config) {
  try {
    const auto model_path = config.value("modelPath", "");
    m_face_score_threshold =
        config.value("scoreThreshold", m_face_score_threshold);
    m_nms_threshold = config.value("nmsThreshold", m_nms_threshold);
    m_top_k = config.value("topK", m_top_k);

    if (model_path.empty()) {
      SPDLOG_ERROR("'modelPath' is missing in config");
      return false;
    }

    m_detector = cv::FaceDetectorYN::create(
        model_path, "", cv::Size(1920, 1080), m_face_score_threshold,
        m_nms_threshold, m_top_k, cv::dnn::DNN_BACKEND_OPENCV,
        cv::dnn::DNN_TARGET_OPENCL);

    SPDLOG_INFO("model_path: %s, score_threshold: %f, top_k: %d",
                model_path.c_str(), m_face_score_threshold, m_top_k);
    return true;
  } catch (const std::exception &e) {
    SPDLOG_ERROR("e.what(): %s", e.what());
    return false;
  }
}

SynchronousProcessingResult YuNetDetect::process(cv::cuda::GpuMat &frame,
                                                 PipelineContext &ctx) {
  ctx.yunet_sface.results.clear();

  cv::cuda::GpuMat frame_to_download;

  // Use the class member for resizing to stop VRAM fragmentation
  if (frame.cols != 1920 || frame.rows != 1080) {
    cv::cuda::resize(frame, m_resized_frame_gpu, cv::Size(1920, 1080));
    frame_to_download = m_resized_frame_gpu;
  } else {
    frame_to_download = frame;
  }

  ctx.yunet_sface.yunet_input_frame_size = frame_to_download.size();
  frame_to_download.download(m_pinned_buffer);

  // Local header to prevent CoW pinning loops
  auto frame_cpu = m_pinned_buffer.createMatHeader();

  cv::Mat faces;
  m_detector->detect(frame_cpu, faces);

  if (!faces.empty()) {
    for (int i = 0; i < faces.rows; ++i) {
      YuNetDetection detection;
      detection.yunet_output =
          faces.row(i).clone(); // Clone prevents downstream leaks
      detection.bounding_box =
          cv::Rect2f(faces.at<float>(i, 0), faces.at<float>(i, 1),
                     faces.at<float>(i, 2), faces.at<float>(i, 3));
      for (int j = 0; j < 5; ++j) {
        detection.landmarks[j] = {faces.at<float>(i, 4 + j * 2),
                                  faces.at<float>(i, 5 + j * 2)};
      }
      detection.face_score = faces.at<float>(i, 14);

      YuNetSFaceResult res;
      res.detection = std::move(detection);
      ctx.yunet_sface.results.push_back(std::move(res));
    }
  }
  return success_and_continue;
}

} // namespace MatrixPipeline::ProcessingUnit

// ============================================================================
// 3. ISOLATED LEAK TESTER
// ============================================================================

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <path_to_face_detection_yunet.onnx> [video_source]\n";
    return -1;
  }
  std::cout << cv::getBuildInformation() << "\n";
  std::cout << "OpenCL Available: " << (cv::ocl::haveOpenCL() ? "Yes" : "No")
            << "\n";
  njson config;
  config.modelPath = argv[1];

  MatrixPipeline::ProcessingUnit::YuNetDetect yunet;
  if (!yunet.init(config)) {
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

  cv::Mat frame;
  cv::cuda::GpuMat d_frame;
  PipelineContext ctx;

  std::cout << "Starting headless stress test. Press Ctrl+C to stop.\n";
  std::cout << "Monitor RAM usage (e.g., using 'htop' or 'nvidia-smi').\n";

  uint64_t frame_count = 0;
  auto start_time = std::chrono::steady_clock::now();

  while (true) {
    cap >> frame;
    if (frame.empty()) {
      // Loop the video if it ends, to keep stress-testing indefinitely
      cap.set(cv::CAP_PROP_POS_FRAMES, 0);
      continue;
    }

    d_frame.upload(frame);

    // The only thing we are testing
    yunet.process(d_frame, ctx);

    frame_count++;
    if (frame_count % 500 == 0) {
      auto now = std::chrono::steady_clock::now();
      auto elapsed =
          std::chrono::duration_cast<std::chrono::seconds>(now - start_time)
              .count();
      std::cout << "Processed " << frame_count << " frames in " << elapsed
                << " seconds. Context results size: "
                << ctx.yunet_sface.results.size() << "\n";
    }
  }

  return 0;
}