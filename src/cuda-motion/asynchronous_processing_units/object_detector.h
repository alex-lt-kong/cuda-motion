#pragma once

#include "../interfaces/i_asynchronous_processing_unit.h"
#include "../interfaces/i_synchronous_processing_unit.h"
#include "http_service.h"

#include <opencv2/cudaarithm.hpp>

#include <vector>
#include <string>
#include <ctime>

namespace CudaMotion::ProcessingUnit {

class ObjectDetector final : public IAsynchronousProcessingUnit {
public:
  ~ObjectDetector() override {
    m_http_service.stop();
  }

  bool init(const njson &config) override {
    try {
      // 1. Init internal Inference Unit
      if (!m_detector.init(config)) return false;

      // 2. Init internal Http Service
      if (config.contains("httpService")) {
          if (!m_http_service.init(config["httpService"])) return false;
          m_http_service.start();
      }

      initialize_resources();
      return true;
    } catch (const std::exception &e) {
      SPDLOG_ERROR("ObjectDetector Init failed: {}", e.what());
      return false;
    }
  }

protected:
  void on_frame_ready(cv::cuda::GpuMat &frame, PipelineContext &ctx) override {
    // 1. Throttling
    if (ctx.frame_seq_num % 30 != 0) return;

    // 2. Run Generic Inference
    // This populates ctx.inference_outputs
    m_detector.process(frame, ctx);


    if (!ctx.indices.empty()) {
      draw_overlay(frame, ctx.indices, ctx.boxes, ctx.class_ids, ctx.confidences);
    }
    // 4. Stream
    m_http_service.enqueue(frame, ctx);
  }

private:
  DetectObjects m_detector;
  HttpService m_http_service;

  std::vector<std::string> m_class_names;
  std::vector<cv::Scalar> m_colors;

  // Reusable Drawing Buffers
  cv::Mat h_overlay_canvas;
  cv::cuda::GpuMat d_overlay_canvas;
  cv::cuda::GpuMat d_overlay_gray;
  cv::cuda::GpuMat d_overlay_mask;

  void initialize_resources() {
      // ... (Same class name & color initialization as before) ...
      m_class_names = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant" /* ... truncated for brevity ... */ };
      std::srand(static_cast<unsigned int>(std::time(nullptr)));
      for (int i = 0; i < 80; i++) m_colors.emplace_back(std::rand() % 255, std::rand() % 255, std::rand() % 255);
  }


  void draw_overlay(cv::cuda::GpuMat &frame,
                    const std::vector<int>& indices,
                    const std::vector<cv::Rect>& boxes,
                    const std::vector<int>& class_ids,
                    const std::vector<float>& confidences) {

      // 1. Prepare CPU Canvas
      if (h_overlay_canvas.size() != frame.size() || h_overlay_canvas.type() != frame.type()) {
          h_overlay_canvas.create(frame.size(), frame.type());
      }
      h_overlay_canvas.setTo(cv::Scalar::all(0));

      // 2. Draw
      for (int idx : indices) {
          int class_id = class_ids[idx];
          if (class_id > 10) continue; // Filter specific classes

          const auto &box = boxes[idx];

          std::string label = (class_id >= 0 && class_id < m_class_names.size()) ? m_class_names[class_id] : "Unknown";
          std::string label_text = fmt::format("{} {:.2f}", label, confidences[idx]);
          cv::Scalar color = (class_id >= 0 && class_id < m_colors.size()) ? m_colors[class_id] : cv::Scalar(0,255,0);

          cv::rectangle(h_overlay_canvas, box, color, 2);

          int baseLine;
          cv::Size labelSize = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
          int top = std::max(box.y, labelSize.height);
          cv::rectangle(h_overlay_canvas, cv::Point(box.x, top - labelSize.height), cv::Point(box.x + labelSize.width, top + baseLine), color, cv::FILLED);
          cv::putText(h_overlay_canvas, label_text, cv::Point(box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
      }

      // 3. Upload & Stamp
      d_overlay_canvas.upload(h_overlay_canvas);
      if (d_overlay_canvas.channels() > 1) {
          cv::cuda::cvtColor(d_overlay_canvas, d_overlay_gray, cv::COLOR_BGR2GRAY);
      } else {
          d_overlay_gray = d_overlay_canvas;
      }
      cv::cuda::threshold(d_overlay_gray, d_overlay_mask, 1, 255, cv::THRESH_BINARY);
      d_overlay_canvas.copyTo(frame, d_overlay_mask);
  }
};

} // namespace CudaMotion::ProcessingUnit