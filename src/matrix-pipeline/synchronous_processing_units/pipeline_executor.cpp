#include "pipeline_executor.h"

#include "../asynchronous_processing_units/asynchronous_processing_unit.h"
#include "../asynchronous_processing_units/http_service.h"
#include "../asynchronous_processing_units/matrix_notifier.h"
#include "../asynchronous_processing_units/rtsp_producer.h"
#include "../asynchronous_processing_units/video_writer.h"
#include "../asynchronous_processing_units/zeromq_publisher.h"
#include "../entities/processing_context.h"
#include "../entities/processing_units_variant.h"
#include "../interfaces/i_synchronous_processing_unit.h"
#include "collect_stats.h"
#include "crop_frame.h"
#include "detect_objects.h"
#include "measure_latency.h"
#include "overlay_bounding_boxes.h"
#include "overlay_info.h"
#include "prune_object_detection_results.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

using njson = nlohmann::json;

namespace MatrixPipeline::ProcessingUnit {

bool PipelineExecutor::init(const njson &settings) {
  using namespace ProcessingUnit;
  SPDLOG_INFO("settings.dump(): {}", settings.dump());
  const auto &settings_pipeline = settings["pipeline"];
  for (nlohmann::basic_json<>::size_type i = 0; i < settings_pipeline.size();
       ++i) {
    ProcessingUnitVariant ptr;
    const std::string type = settings_pipeline[i]["type"].get<std::string>();
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
    } else if (type == "SynchronousProcessingUnit::collectStats") {
      ptr = std::make_unique<CollectStats>();
    } else if (type == "SynchronousProcessingUnit::measureLatency") {
      ptr = std::make_unique<MeasureLatency>();
    } else if (type ==
               "SynchronousProcessingUnit::pruneObjectDetectionResults") {
      ptr = std::make_unique<PruneObjectDetectionResults>();
    } else if (type == "AsynchronousProcessingUnit::videoWriter") {
      ptr = std::make_unique<VideoWriter>();
    } else if (type == "AsynchronousProcessingUnit::rtspProducer") {
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
    SPDLOG_INFO("Adding {}-th processing unit, type: {}", i, type);
    if (std::visit(
            overload{
                [&](const std::unique_ptr<ISynchronousProcessingUnit> &ptr_) {
                  return ptr_->init(settings_pipeline[i]);
                },
                [&](const std::unique_ptr<IAsynchronousProcessingUnit> &ptr_) {
                  if (!ptr_->init(settings_pipeline[i])) {
                    SPDLOG_ERROR("returned false!");
                    return false;
                  }
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

SynchronousProcessingResult PipelineExecutor::process(cv::cuda::GpuMat &frame,
                                                      PipelineContext &ctx) {
  for (size_t i = 0; i < m_processing_units.size(); ++i) {
    ctx.processing_unit_idx = i;
    const auto retval = std::visit(
        overload{
            [&](const std::unique_ptr<ISynchronousProcessingUnit> &ptr) {
              return ptr->process(frame, ctx);
            },
            [&](const std::unique_ptr<IAsynchronousProcessingUnit> &ptr) {
              return ptr->enqueue(frame, ctx);
            },
        },
        m_processing_units[i]);
    if (retval == failure_and_stop || retval == success_and_stop)
      break;
  }
}

;
} // namespace MatrixPipeline::ProcessingUnit
