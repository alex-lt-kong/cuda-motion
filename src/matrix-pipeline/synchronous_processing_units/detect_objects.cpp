#include "../interfaces/i_synchronous_processing_unit.h"

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/dnn.hpp>
#include <spdlog/spdlog.h>

#include <string>
#include <vector>

namespace CudaMotion::ProcessingUnit {

bool DetectObjects::init(const njson &config) {
  try {
    if (!config.contains("modelPath")) {
      SPDLOG_ERROR("modelPath not defined");
      return false;
    }

    m_model_path = config["modelPath"].get<std::string>();

    // Allow overriding input size via config
    if (config.contains("inputWidth"))
      m_model_input_size.width = config["inputWidth"].get<int>();
    if (config.contains("inputHeight"))
      m_model_input_size.height = config["inputHeight"].get<int>();
    if (config.contains("frameInterval"))
      m_frame_interval = config["frameInterval"].get<int>();

    SPDLOG_INFO("Loading ONNX model: {}", m_model_path);

    m_net = cv::dnn::readNetFromONNX(m_model_path);
    m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    return !m_net.empty();
  } catch (const std::exception &e) {
    SPDLOG_ERROR("Init failed: {}", e.what());
    return false;
  }
}

void DetectObjects::post_process_yolo(const cv::cuda::GpuMat &frame,
                                      PipelineContext &ctx) const {
  const std::vector<cv::Mat> &outputs = ctx.yolo.inference_outputs;

  // Calculate Scale Factor (Frame vs Input Blob)
  double x_factor =
      static_cast<double>(frame.cols) / ctx.yolo.inference_input_size.width;
  double y_factor =
      static_cast<double>(frame.rows) / ctx.yolo.inference_input_size.height;

  // --- YOLO Output Parsing ---
  // (This logic is specific to YOLOv5/v8 flattened output)
  const cv::Mat &output = outputs[0];
  int dimensions = output.size[1];
  int rows = output.size[2];

  cv::Mat result_wrapper(dimensions, rows, CV_32F, output.data);
  cv::Mat output_t;
  cv::transpose(result_wrapper, output_t); // Transpose to [rows, dimensions]

  ctx.yolo.class_ids.clear();
  ctx.yolo.confidences.clear();
  ctx.yolo.boxes.clear();

  for (int i = 0; i < rows; ++i) {
    auto *row_ptr = output_t.ptr<float>(i);
    cv::Mat scores(1, dimensions - 4, CV_32F, row_ptr + 4);
    cv::Point class_id_point;
    double max_class_score;
    cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);

    if (max_class_score > m_conf_thres) {
      float cx = row_ptr[0];
      float cy = row_ptr[1];
      float w = row_ptr[2];
      float h = row_ptr[3];

      int left = int((cx - 0.5 * w) * x_factor);
      int top = int((cy - 0.5 * h) * y_factor);
      int width = int(w * x_factor);
      int height = int(h * y_factor);

      ctx.yolo.boxes.emplace_back(left, top, width, height);
      ctx.yolo.confidences.push_back(static_cast<float>(max_class_score));
      ctx.yolo.class_ids.push_back(class_id_point.x);
    }
  }

  ctx.yolo.indices.clear();
  cv::dnn::NMSBoxes(ctx.yolo.boxes, ctx.yolo.confidences, m_conf_thres,
                    m_nms_thres, ctx.yolo.indices);
}

SynchronousProcessingResult DetectObjects::process(cv::cuda::GpuMat &frame,
                                                   PipelineContext &ctx) {
  if (ctx.frame_seq_num % m_frame_interval != 0) {
    ctx.yolo = m_prev_yolo_ctx;
    return success_and_continue;
  }

  if (frame.empty() || m_net.empty())
    return failure_and_continue;

  // Clear previous results
  ctx.yolo.inference_outputs.clear();
  ctx.yolo.inference_input_size = m_model_input_size;

  try {
    // 1. Pre-process: Resize on GPU
    cv::cuda::GpuMat resized_frame;
    cv::cuda::resize(frame, resized_frame, m_model_input_size);

    // YOLO expects RGB (OpenCV default is BGR)
    cv::cuda::cvtColor(resized_frame, resized_frame, cv::COLOR_BGR2RGB);

    // 2. Pre-process: Create Blob (Requires CPU transfer currently)
    cv::Mat cpu_small_frame;
    resized_frame.download(cpu_small_frame);

    cv::Mat blob;
    cv::dnn::blobFromImage(cpu_small_frame, blob, 1.0 / 255.0, cv::Size(),
                           cv::Scalar(), false, false);

    // 3. Inference
    m_net.setInput(blob);

    // Populate the vector in the context directly
    m_net.forward(ctx.yolo.inference_outputs,
                  m_net.getUnconnectedOutLayersNames());

    // 3. Decode & Visualize (If inference produced data)
    if (!ctx.yolo.inference_outputs.empty()) {
      post_process_yolo(frame, ctx);
    }
    m_prev_yolo_ctx = ctx.yolo;
    return success_and_continue;

  } catch (const cv::Exception &e) {
    SPDLOG_ERROR("Inference Error: {}", e.what());
    return failure_and_continue;
  }
}

} // namespace CudaMotion::ProcessingUnit