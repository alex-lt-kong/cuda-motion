#pragma once

#include "asynchronous_processing_units/http_service.h"
#include "asynchronous_processing_units/rtsp_producer.h"
#include "asynchronous_processing_units/video_writer.h"
#include "asynchronous_processing_units/zeromq_publisher.h"
#include "entities/processing_context.h"
#include "entities/processing_units_variant.h"
#include "interfaces/i_synchronous_processing_unit.h"
#include "synchronous_processing_units/annotate_bounding_boxes.h"
#include "synchronous_processing_units/calculate_change_rate.h"
#include "synchronous_processing_units/control_fps.h"
#include "synchronous_processing_units/crop_frame.h"
#include "synchronous_processing_units/detect_objects.h"
#include "synchronous_processing_units/measure_latency.h"
#include "synchronous_processing_units/overlay_bounding_boxes.h"
#include "synchronous_processing_units/overlay_info.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

using njson = nlohmann::json;

namespace MatrixPipeline {
class PipelineExecutor {
private:
  /**
   *
   */
  std::vector<ProcessingUnit::ProcessingUnitVariant> m_processing_units;

public:
  PipelineExecutor() = default;

  ~PipelineExecutor() = default;

  bool init(const njson &settings) {
    using namespace ProcessingUnit;
    for (nlohmann::basic_json<>::size_type i = 0; i < settings.size(); ++i) {
      ProcessingUnitVariant ptr;
      const std::string type = settings[i]["type"].get<std::string>();
      if (type == "SynchronousProcessingUnit::rotation") {
        ptr = std::make_unique<RotateFrame>();
      } else if (type == "SynchronousProcessingUnit::overlayInfo") {
        ptr = std::make_unique<OverlayInfo>();
      } else if (type == "SynchronousProcessingUnit::cropFrame") {
        ptr = std::make_unique<CropFrame>();
      } else if (type == "SynchronousProcessingUnit::debugOutput") {
        ptr = std::make_unique<DebugOutput>();
      } else if (type == "SynchronousProcessingUnit::resizeFrame") {
        ptr = std::make_unique<ResizeFrame>();
      } else if (type == "SynchronousProcessingUnit::calculateChangeRate") {
        ptr = std::make_unique<CalculateChangeRate>();
      } else if (type == "SynchronousProcessingUnit::controlFps") {
        ptr = std::make_unique<ControlFps>();
      } else if (type == "SynchronousProcessingUnit::measureLatency") {
        ptr = std::make_unique<MeasureLatency>();
      } else if (type == "SynchronousProcessingUnit::annotateBoundingBoxes") {
        ptr = std::make_unique<AnnotateBoundingBoxes>();
      } else if (type == "AsynchronousProcessingUnit::videoWriter") {
        ptr = std::make_unique<VideoWriter>();
      }else if (type == "AsynchronousProcessingUnit::rtspProducer") {
        ptr = std::make_unique<RtspProducer>();
      } else if (type == "AsynchronousProcessingUnit::httpService") {
        ptr = std::make_unique<HttpService>();
      } else if (type == "SynchronousProcessingUnit::detectObjects") {
        ptr = std::make_unique<DetectObjects>();
      } else if (type == "SynchronousProcessingUnit::overlayBoundingBoxes") {
        ptr = std::make_unique<OverlayBoundingBoxes>();
      } else if (type ==
                 "AsynchronousProcessingUnit::asynchronousProcessingUnit") {
        ptr = std::make_unique<AsynchronousProcessingUnit>();
      } else if (type == "AsynchronousProcessingUnit::matrixNotifier") {
        ptr = std::make_unique<MatrixNotifier>();
      } else {
        SPDLOG_WARN("Unrecognized pipeline unit, type: {}, idx: {}", type, i);
        continue;
      }
      if (std::visit(
              overload{
                  [&](const std::unique_ptr<ISynchronousProcessingUnit> &ptr_) {
                    return ptr_->init(settings[i]);
                  },
                  [&](const std::unique_ptr<IAsynchronousProcessingUnit>
                          &ptr_) {
                    if (!ptr_->init(settings[i]))
                      return false;
                    ptr_->start();
                    return true;
                  },
              },
              ptr)) {
        m_processing_units.push_back(std::move(ptr));
        SPDLOG_INFO("Added {}-th processing unit, type: {}", i, type);
      } else {
        SPDLOG_WARN("not added");
      }
    }
    return true;
  }

  void on_frame_ready(cv::cuda::GpuMat &frame,
                      ProcessingUnit::PipelineContext &ctx) {
    for (size_t i = 0; i < m_processing_units.size(); ++i) {
      ctx.processing_unit_idx = i;
      const auto retval = std::visit(
          ProcessingUnit::overload{
              [&](const std::unique_ptr<
                  ProcessingUnit::ISynchronousProcessingUnit> &ptr) {
                return ptr->process(frame, ctx);
              },
              [&](const std::unique_ptr<
                  ProcessingUnit::IAsynchronousProcessingUnit> &ptr) {
                return ptr->enqueue(frame, ctx);
              },
          },
          m_processing_units[i]);
      if (retval == ProcessingUnit::failure_and_stop ||
          retval == ProcessingUnit::success_and_stop)
        break;
    }
  }
};
} // namespace MatrixPipeline