#pragma once

#include "asynchronous_processing_units/http_service.h"
#include "asynchronous_processing_units/video_writer.h"
#include "asynchronous_processing_units/zeromq_publisher.h"
#include "entities/processing_context.h"
#include "entities/processing_units_variant.h"
#include "interfaces/i_synchronous_processing_unit.h"
#include "synchronous_processing_units/calculate_change_rate.h"
#include "synchronous_processing_units/control_fps.h"
#include "synchronous_processing_units/crop_frame.h"
#include "synchronous_processing_units/overlay_info.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

using njson = nlohmann::json;

namespace CudaMotion {
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
      if (settings[i]["type"].get<std::string>() ==
          "SynchronousProcessingUnit::rotation") {
        ptr = std::make_unique<RotateFrame>();
      } else if (settings[i]["type"].get<std::string>() ==
                 "SynchronousProcessingUnit::overlayInfo") {
        ptr = std::make_unique<OverlayInfo>();
      } else if (settings[i]["type"].get<std::string>() ==
                 "SynchronousProcessingUnit::cropFrame") {
        ptr = std::make_unique<CropFrame>();
      } else if (settings[i]["type"].get<std::string>() ==
                 "SynchronousProcessingUnit::debugOutput") {
        ptr = std::make_unique<DebugOutput>();
      } else if (settings[i]["type"].get<std::string>() ==
                 "SynchronousProcessingUnit::resizeFrame") {
        ptr = std::make_unique<ResizeFrame>();
      } else if (settings[i]["type"].get<std::string>() ==
                 "SynchronousProcessingUnit::calculateChangeRate") {
        ptr = std::make_unique<CalculateChangeRate>();
      } else if (settings[i]["type"].get<std::string>() ==
                 "SynchronousProcessingUnit::controlFps") {
        ptr = std::make_unique<ControlFps>();
      } else if (settings[i]["type"].get<std::string>() ==
                 "AsynchronousProcessingUnit::videoWriter") {
        ptr = std::make_unique<VideoWriter>();
      } else if (settings[i]["type"].get<std::string>() ==
                 "AsynchronousProcessingUnit::httpService") {
        ptr = std::make_unique<HttpService>();
      } else if (settings[i]["type"].get<std::string>() ==
                 "SynchronousProcessingUnit::detectObjects") {
        ptr = std::make_unique<DetectObjects>();
      } else if (settings[i]["type"].get<std::string>() ==
                 "SynchronousProcessingUnit::overlayBoundingBoxes") {
        ptr = std::make_unique<OverlayBoundingBoxes>();
      } else if (settings[i]["type"].get<std::string>() ==
                 "AsynchronousProcessingUnit::asynchronousProcessingUnit") {
        ptr = std::make_unique<AsynchronousProcessingUnit>();
      } else if (settings[i]["type"].get<std::string>() ==
                 "AsynchronousProcessingUnit::matrixNotifier") {
        ptr = std::make_unique<MatrixNotifier>();
      } else {
        SPDLOG_WARN("Unrecognized pipeline unit: {}",
                    settings[i]["type"].get<std::string>());
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
              ptr))
        m_processing_units.push_back(std::move(ptr));
      else {
        SPDLOG_WARN("not added");
      }
    }
    return true;
  }

  void on_frame_ready(cv::cuda::GpuMat &frame,
                      ProcessingUnit::PipelineContext &ctx) {
    for (size_t i = 0; i < m_processing_units.size(); ++i) {
      ctx.processing_unit_idx = i;
      auto retval = std::visit(
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
} // namespace CudaMotion