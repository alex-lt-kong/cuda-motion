#include "asynchronous_processing_unit.h"
#include "../asynchronous_processing_units/http_service.h"
#include "../asynchronous_processing_units/matrix_notifier.h"
#include "../asynchronous_processing_units/rtsp_producer.h"
#include "../asynchronous_processing_units/video_writer.h"
#include "../asynchronous_processing_units/zeromq_publisher.h"
#include "../entities/processing_context.h"
#include "../entities/processing_units_variant.h"
#include "../interfaces/i_synchronous_processing_unit.h"
#include "../synchronous_processing_units/collect_stats.h"
#include "../synchronous_processing_units/rotate_frame.h"
#include "../synchronous_processing_units/crop_frame.h"
#include "../synchronous_processing_units/measure_latency.h"
#include "../synchronous_processing_units/overlay_info.h"
#include "../synchronous_processing_units/resize_frame.h"
#include "../synchronous_processing_units/yolo_detect.h"
#include "../synchronous_processing_units/yolo_overlay_bounding_boxes.h"
#include "../synchronous_processing_units/yolo_prune_detection_results.h"

#include <iostream>

namespace MatrixPipeline::ProcessingUnit {

bool AsynchronousProcessingUnit::init(const njson &config) {
  // m_exe = std::make_unique<PipelineExecutor>();

  using namespace ProcessingUnit;
  SPDLOG_INFO("settings.dump(): {}", config.dump());
  const auto &settings_pipeline = config["pipeline"];
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
                   "SynchronousProcessingUnit::pruneObjectDetectionResults" ||
               type == "SynchronousProcessingUnit::yoloPruneDetectionResults") {
      ptr = std::make_unique<YoloPruneDetectionResults>();
    } else if (type == "AsynchronousProcessingUnit::videoWriter") {
      ptr = std::make_unique<VideoWriter>();
    } else if (type == "AsynchronousProcessingUnit::rtspProducer") {
      ptr = std::make_unique<RtspProducer>();
    } else if (type == "AsynchronousProcessingUnit::httpService") {
      ptr = std::make_unique<HttpService>();
    } else if (type == "SynchronousProcessingUnit::detectObjects" ||
               type == "SynchronousProcessingUnit::yoloDetect") {
      ptr = std::make_unique<YoloDetect>();
    } else if (type == "SynchronousProcessingUnit::overlayBoundingBoxes" ||
               type == "SynchronousProcessingUnit::yoloOverlayBoundingBoxes") {
      ptr = std::make_unique<YoloOverlayBoundingBoxes>();
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
                  if (!ptr_->init(settings_pipeline[i]))
                    return false;
                  ptr_->start();
                  return true;
                },
            },
            ptr)) {
      m_processing_units.push_back(std::move(ptr));
      SPDLOG_INFO("Added {}-th processing unit", i);
    } else {
      SPDLOG_ERROR("NOT added {}-th processing unit", i);
    }
  }
  return true;
}

void AsynchronousProcessingUnit::on_frame_ready(cv::cuda::GpuMat &frame,
                                                PipelineContext &ctx) {
  // std::lock_guard g(m_mutex);
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

} // namespace MatrixPipeline::ProcessingUnit