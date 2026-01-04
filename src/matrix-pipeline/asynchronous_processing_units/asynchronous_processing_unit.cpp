#include "asynchronous_processing_unit.h"

#include "../asynchronous_processing_units/http_service.h"
#include "../asynchronous_processing_units/matrix_notifier.h"
#include "../asynchronous_processing_units/video_writer.h"
#include "../asynchronous_processing_units/zeromq_publisher.h"
#include "../entities/processing_context.h"
#include "../entities/processing_units_variant.h"
#include "../interfaces/i_synchronous_processing_unit.h"
#include "../synchronous_processing_units/collect_stats.h"
#include "../synchronous_processing_units/crop_frame.h"
#include "../synchronous_processing_units/debug_output.h"
#include "../synchronous_processing_units/measure_latency.h"
#include "../synchronous_processing_units/overlay_info.h"
#include "../synchronous_processing_units/resize_frame.h"
#include "../synchronous_processing_units/rotate_frame.h"
#include "../synchronous_processing_units/yolo_detect.h"
#include "../synchronous_processing_units/yolo_overlay_bounding_boxes.h"
#include "../synchronous_processing_units/yolo_prune_detection_results.h"
#include "../synchronous_processing_units/yunet_detect.h"
#include "../synchronous_processing_units/yunet_overlay_landmarks.h"
#include "ffmpeg_streamer.h"

#include <fmt/ranges.h>

namespace MatrixPipeline::ProcessingUnit {

bool AsynchronousProcessingUnit::init(const njson &config) {
  // m_exe = std::make_unique<PipelineExecutor>();

  using namespace ProcessingUnit;
  SPDLOG_INFO("settings.dump(): {}", config.dump());
  const auto &settings_pipeline = config["pipeline"];
  m_turned_on_hours = config.value("turnedOnHours", m_turned_on_hours);
  for (nlohmann::basic_json<>::size_type i = 0; i < settings_pipeline.size();
       ++i) {
    try {
      ProcessingUnitVariant ptr;
      const std::string type = settings_pipeline[i]["type"].get<std::string>();
      if (type == "SynchronousProcessingUnit::rotation") {
        ptr = std::make_unique<RotateFrame>(m_unit_path);
      } else if (type == "SynchronousProcessingUnit::overlayInfo") {
        ptr = std::make_unique<OverlayInfo>(m_unit_path);
      } else if (type == "SynchronousProcessingUnit::cropFrame") {
        ptr = std::make_unique<CropFrame>(m_unit_path);
      } else if (type == "SynchronousProcessingUnit::debugOutput") {
        ptr = std::make_unique<DebugOutput>(m_unit_path);
      } else if (type == "SynchronousProcessingUnit::resizeFrame") {
        ptr = std::make_unique<ResizeFrame>(m_unit_path);
      } else if (type == "SynchronousProcessingUnit::collectStats") {
        ptr = std::make_unique<CollectStats>(m_unit_path);
      } else if (type == "SynchronousProcessingUnit::measureLatency") {
        ptr = std::make_unique<MeasureLatency>(m_unit_path);
      } else if (type ==
                 "SynchronousProcessingUnit::yoloPruneDetectionResults") {
        ptr = std::make_unique<YoloPruneDetectionResults>(m_unit_path);
      } else if (type == "AsynchronousProcessingUnit::videoWriter") {
        ptr = std::make_unique<VideoWriter>(m_unit_path);
      } else if (type == "AsynchronousProcessingUnit::httpService") {
        ptr = std::make_unique<HttpService>(m_unit_path);
      } else if (type == "SynchronousProcessingUnit::yoloDetect") {
        ptr = std::make_unique<YoloDetect>(m_unit_path);
      } else if (type == "SynchronousProcessingUnit::yuNetDetect") {
        ptr = std::make_unique<YuNetDetect>(m_unit_path);
      } else if (type == "SynchronousProcessingUnit::yuNetOverlayLandmarks") {
        ptr = std::make_unique<YuNetOverlayLandmarks>(m_unit_path);
      } else if (type ==
                 "SynchronousProcessingUnit::yoloOverlayBoundingBoxes") {
        ptr = std::make_unique<YoloOverlayBoundingBoxes>(m_unit_path);
      } else if (type ==
                 "AsynchronousProcessingUnit::asynchronousProcessingUnit") {
        ptr = std::make_unique<AsynchronousProcessingUnit>(m_unit_path);
      } else if (type == "AsynchronousProcessingUnit::matrixNotifier") {
        ptr = std::make_unique<MatrixNotifier>(m_unit_path);
      } else if (type == "AsynchronousProcessingUnit::ffmpegStreamerUnit") {
        ptr = std::make_unique<FFmpegStreamerUnit>(m_unit_path);
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
                  [&](const std::unique_ptr<IAsynchronousProcessingUnit>
                          &ptr_) {
                    if (!ptr_->init(settings_pipeline[i]))
                      return false;
                    ptr_->start();
                    return true;
                  },
              },
              ptr)) {
        m_processing_units.push_back(std::move(ptr));
        SPDLOG_INFO("Added {}-th processing unit, turned_on_hours: {}", i,
                    fmt::join(m_turned_on_hours, ","));
      } else {
        SPDLOG_ERROR("NOT added {}-th processing unit", i);
      }
    } catch (const std::exception &e) {
      SPDLOG_ERROR("{}-th processing unit's init() throws exception and thus "
                   "skipped. e.what(): {}'",
                   i, e.what());
      return false;
    }
  }
  return true;
}

void AsynchronousProcessingUnit::on_frame_ready(cv::cuda::GpuMat &frame,
                                                PipelineContext &ctx) {
  for (size_t i = 0; i < m_processing_units.size(); ++i) {
    ctx.processing_unit_idx = i;
    const auto retval = std::visit(
        overload{
            [&](const std::unique_ptr<ISynchronousProcessingUnit> &ptr) {
              if (ptr->is_disabled())
                return failure_and_continue;
              return ptr->process(frame, ctx);
            },
            [&](const std::unique_ptr<IAsynchronousProcessingUnit> &ptr) {
              if (ptr->is_disabled())
                return failure_and_continue;
              return ptr->enqueue(frame, ctx);
            },
        },
        m_processing_units[i]);
    if (retval == failure_and_stop || retval == success_and_stop)
      break;
  }
}

} // namespace MatrixPipeline::ProcessingUnit