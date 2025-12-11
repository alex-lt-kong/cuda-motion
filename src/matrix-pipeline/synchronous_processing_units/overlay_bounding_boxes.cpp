#include "overlay_bounding_boxes.h"

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>

#include <ctime>
#include <algorithm>

namespace MatrixPipeline::ProcessingUnit {

bool OverlayBoundingBoxes::init([[maybe_unused]]const njson &config) {
    // 1. Initialize Class Names (Standard COCO 80 classes)
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

    // 2. Initialize Random Colors
    // We seed with time to get different colors on different runs, 
    // or you can fix the seed for consistency.
    std::srand(1); // we want deterministic coloring
    m_colors.clear();
    m_colors.reserve(80);
    for (int i = 0; i < 80; i++) {
        m_colors.emplace_back(std::rand() % 255, std::rand() % 255, std::rand() % 255);
    }

    return true;
}

SynchronousProcessingResult OverlayBoundingBoxes::process(cv::cuda::GpuMat &frame, PipelineContext& ctx) {
    // Fast exit if no detections or invalid frame
    if (frame.empty() || ctx.yolo.indices.empty()) {
        return success_and_continue;
    }

    try {
        // 1. Prepare CPU Canvas
        // Ensure CPU canvas matches the GPU frame size/type
      if (h_overlay_canvas.size() != frame.size() ||
          h_overlay_canvas.type() != frame.type()) {
        h_overlay_canvas.create(frame.size(), frame.type());
      }

      // Clear canvas to black (0,0,0) - this is our transparent key
      h_overlay_canvas.setTo(cv::Scalar::all(0));

      // 2. Draw Detections on CPU Canvas
      for (int idx : ctx.yolo.indices) {
        int class_id = ctx.yolo.class_ids[idx];

        // Safety Checks
        if (class_id < 0 ||
            static_cast<size_t>(class_id) >= m_class_names.size())
          continue;

        // Optional: Hardcoded filter from original requirements
        // (Person/Vehicles only)
        if (class_id > 10)
          continue;

        const auto &box = ctx.yolo.boxes[idx];
        float conf = ctx.yolo.confidences[idx];

        std::string label = m_class_names[class_id];
        std::string label_text = fmt::format("{}{} {:.2f} ", !ctx.yolo.is_in_roi[idx] ? "(!)" : "", label, conf);

        cv::Scalar color;
        if (!ctx.yolo.is_in_roi[idx])
          color = cv::Scalar(127, 127, 127);
        else
          color = (static_cast<size_t>(class_id) < m_colors.size())
                      ? m_colors[class_id]
                      : cv::Scalar(0, 255, 0);

        // Draw Bounding Box
        cv::rectangle(h_overlay_canvas, box, color, 2);

        // Draw Label Background
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            int top = std::max(box.y, labelSize.height);
            
            cv::rectangle(h_overlay_canvas, 
                          cv::Point(box.x, top - labelSize.height),
                          cv::Point(box.x + labelSize.width, top + baseLine), 
                          color, cv::FILLED);
            
            // Draw Label Text (White)
            cv::putText(h_overlay_canvas, label_text, cv::Point(box.x, top),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }

        // 3. Upload Canvas to GPU
        d_overlay_canvas.upload(h_overlay_canvas);

        // 4. Generate Mask from Canvas
        // Convert to grayscale to check for non-black pixels
        if (d_overlay_canvas.channels() > 1) {
            cv::cuda::cvtColor(d_overlay_canvas, d_overlay_gray, cv::COLOR_BGR2GRAY);
        } else {
            d_overlay_gray = d_overlay_canvas;
        }

        // Threshold: 1 = Drawing Present (Keep Canvas), 0 = Black (Keep Original Frame)
        cv::cuda::threshold(d_overlay_gray, d_overlay_mask, 1, 255, cv::THRESH_BINARY);

        // 5. Stamp Overlay onto Original Frame
        // This copies pixels from d_overlay_canvas into 'frame' ONLY where mask is non-zero.
        d_overlay_canvas.copyTo(frame, d_overlay_mask);

        return success_and_continue;

    } catch (const cv::Exception &e) {
        SPDLOG_ERROR("OverlayBoundingBoxes OpenCV Error: {}", e.what());
        return failure_and_continue;
    }
}

} // namespace MatrixPipeline::ProcessingUnit