#pragma once

#include "../interfaces/i_asynchronous_processing_unit.h"

// OpenCV Dependencies
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

// Standard Libs
#include <algorithm>
#include <string>
#include <vector>

namespace CudaMotion::ProcessingUnit {

class ObjectDetector final : public IAsynchronousProcessingUnit {
public:
  ~ObjectDetector() override {
    m_ms.stop();
    IAsynchronousProcessingUnit::stop();
  }

  bool init(const njson &config) override {
    try {
      if (!config.contains("model_path")) {
        SPDLOG_ERROR("Config missing 'model_path'");
        return false;
      }

      m_model_path = config["model_path"].get<std::string>();
      m_conf_thres = config.value("conf_thres", 0.5f);
      m_nms_thres = config.value("nms_thres", 0.45f);

      // Initialize COCO Class Names
      m_class_names = {
          "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
          "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
          "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
          "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
          "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
          "hair drier", "toothbrush"
      };

      // Generate random colors
      std::srand(0);
      for (int i = 0; i < 80; i++) {
          m_colors.push_back(cv::Scalar(std::rand() % 255, std::rand() % 255, std::rand() % 255));
      }

      SPDLOG_INFO("Loading ONNX: {}", m_model_path);

      m_net = cv::dnn::readNetFromONNX(m_model_path);
      m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
      m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

      njson ms_configs;
      ms_configs["refreshIntervalSec"] = config["httpService"]["refreshIntervalSec"];
      ms_configs["bindAddr"] = config["httpService"]["bindAddr"];
      ms_configs["port"] = config["httpService"]["port"];
      ms_configs["username"] = config["httpService"]["username"];
      ms_configs["password"] = config["httpService"]["password"];
      m_ms.init(ms_configs);
      m_ms.start();
      return true;
    } catch (const std::exception &e) {
      SPDLOG_ERROR("Init failed: {}", e.what());
      return false;
    }
  }

//  void stop() override { IAsynchronousProcessingUnit::stop(); }

protected:
  void on_frame_ready(cv::cuda::GpuMat &frame, PipelineContext &ctx) override {
    if (ctx.frame_seq_num % 30 != 0) return;
    process_frame(frame, ctx);
  }

private:
  // --- Config ---
  std::string m_model_path;
  float m_conf_thres = 0.5f;
  float m_nms_thres = 0.45f;
  HttpService m_ms;
  cv::Size m_model_input_size = {640, 640};

  std::vector<std::string> m_class_names;
  std::vector<cv::Scalar> m_colors;
  cv::dnn::Net m_net;

  // --- Reusable Overlay Buffers (Similar to OverlayInfo) ---
  cv::Mat h_overlay_canvas;       // CPU drawing canvas
  cv::cuda::GpuMat d_overlay_canvas; // GPU copy of canvas
  cv::cuda::GpuMat d_overlay_gray;   // For mask generation
  cv::cuda::GpuMat d_overlay_mask;   // The final mask

  void process_frame(cv::cuda::GpuMat &frame, const PipelineContext &ctx) {
    try {
      float x_scale = (float)frame.cols / m_model_input_size.width;
      float y_scale = (float)frame.rows / m_model_input_size.height;

      // 1. GPU Resize for Inference (Fast)
      cv::cuda::GpuMat gpu_resized;
      cv::cuda::resize(frame, gpu_resized, m_model_input_size);
      cv::cuda::cvtColor(gpu_resized, gpu_resized, cv::COLOR_BGR2RGB);

      // 2. Small Download for Inference (640x640)
      cv::Mat cpu_small_frame;
      gpu_resized.download(cpu_small_frame);

      cv::Mat blob;
      cv::dnn::blobFromImage(cpu_small_frame, blob, 1.0 / 255.0, cv::Size(), cv::Scalar(), false, false);

      // 3. Inference
      m_net.setInput(blob);
      std::vector<cv::Mat> outputs;
      m_net.forward(outputs, m_net.getUnconnectedOutLayersNames());

      // 4. Post-Process & Overlay (Zero-Copy on original frame)
      post_process_and_overlay(frame, outputs, ctx, x_scale, y_scale);
      m_ms.enqueue(frame, ctx);
    } catch (const cv::Exception &e) {
      SPDLOG_ERROR("Inference Error: {}", e.what());
    }
  }

  void post_process_and_overlay(cv::cuda::GpuMat &frame,
                                const std::vector<cv::Mat> &outputs,
                                const PipelineContext &ctx,
                                float x_factor, float y_factor) {
    if (outputs.empty()) return;

    // --- Decode YOLO Output ---
    const cv::Mat &output = outputs[0];
    int dimensions = output.size[1];
    int rows = output.size[2];

    cv::Mat result_wrapper(dimensions, rows, CV_32F, output.data);
    cv::Mat output_t;
    cv::transpose(result_wrapper, output_t);

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i) {
      float *row_ptr = output_t.ptr<float>(i);
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

        boxes.push_back(cv::Rect(left, top, width, height));
        confidences.push_back((float)max_class_score);
        class_ids.push_back(class_id_point.x);
      }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, m_conf_thres, m_nms_thres, indices);

    if (!indices.empty()) {
        SPDLOG_INFO("Frame {} detected {} objects.", ctx.frame_seq_num, indices.size());

        // --- OVERLAY STRATEGY ---

        // 1. Prepare CPU Canvas
        // Ensure canvas size matches the GPU frame
        if (h_overlay_canvas.size() != frame.size() || h_overlay_canvas.type() != frame.type()) {
            h_overlay_canvas.create(frame.size(), frame.type());
        }

        // Clear canvas to Black (0,0,0) efficiently
        h_overlay_canvas.setTo(cv::Scalar::all(0));

        // 2. Draw on CPU Canvas
        for (int idx : indices) {
          if (class_ids[idx] > 10) continue;
            const auto &box = boxes[idx];
            int class_id = class_ids[idx];
            float conf = confidences[idx];

            std::string label = (class_id >= 0 && class_id < m_class_names.size())
                                ? m_class_names[class_id] : "Unknown";
            std::string label_text = fmt::format("{} {:.2f}", label, conf);
            cv::Scalar color = (class_id >= 0 && class_id < m_colors.size())
                                ? m_colors[class_id] : cv::Scalar(0, 255, 0);

            // Draw Box
            cv::rectangle(h_overlay_canvas, box, color, 2);

            // Draw Label
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            int top = std::max(box.y, labelSize.height);
            cv::rectangle(h_overlay_canvas, cv::Point(box.x, top - labelSize.height),
                          cv::Point(box.x + labelSize.width, top + baseLine), color, cv::FILLED);
            cv::putText(h_overlay_canvas, label_text, cv::Point(box.x, top),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }

        // 3. Upload Canvas to GPU (1x Upload)
        d_overlay_canvas.upload(h_overlay_canvas);

        // 4. Create Mask (Where Canvas != Black)
        if (d_overlay_canvas.channels() > 1) {
            cv::cuda::cvtColor(d_overlay_canvas, d_overlay_gray, cv::COLOR_BGR2GRAY);
        } else {
            d_overlay_gray = d_overlay_canvas;
        }

        // Threshold > 1 to create binary mask (1 = overlay, 0 = keep original)
        cv::cuda::threshold(d_overlay_gray, d_overlay_mask, 1, 255, cv::THRESH_BINARY);

        // 5. Apply Overlay using Mask (GPU Operation)
        // Copy d_overlay_canvas onto 'frame' WHERE d_overlay_mask is not zero
        d_overlay_canvas.copyTo(frame, d_overlay_mask);
    }
  }
};

} // namespace CudaMotion::ProcessingUnit