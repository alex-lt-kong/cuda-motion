#include "device_manager.h"
#include "asynchronous_processing_units/http_service.h"
#include "asynchronous_processing_units/video_writer.h"
#include "asynchronous_processing_units/zeromq_publisher.h"
#include "entities/processing_units_variant.h"
#include "global_vars.h"
#include "interfaces/i_synchronous_processing_unit.h"
#include "synchronous_processing_units/calculate_change_rate.h"
#include "synchronous_processing_units/control_fps.h"
#include "synchronous_processing_units/crop_frame.h"
#include "synchronous_processing_units/debug.h"
#include "synchronous_processing_units/overlay_info.h"
#include "synchronous_processing_units/resize_frame.h"
#include "synchronous_processing_units/rotate_frame.h"

#include <cstddef>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/cudacodec.hpp>
#include <spdlog/spdlog.h>

#include <regex>
#include <sys/socket.h>
#include <variant>

using namespace std;

namespace CudaMotion {

DeviceManager::DeviceManager() {}

DeviceManager::~DeviceManager() {}

void DeviceManager::InternalThreadEntry() {
  using namespace ProcessingUnit;
  std::vector<ProcessingUnitVariant> processing_units;
  for (nlohmann::basic_json<>::size_type i = 0; i < settings["pipeline"].size();
       ++i) {
    settings["pipeline"][i]["device"] = settings["device"];
    ProcessingUnitVariant ptr;
    if (settings["pipeline"][i]["type"].get<std::string>() ==
        "SynchronousProcessingUnit::rotation") {
      ptr = std::make_unique<RotateFrame>();
    } else if (settings["pipeline"][i]["type"].get<std::string>() ==
               "SynchronousProcessingUnit::overlayInfo") {
      ptr = std::make_unique<OverlayInfo>();
    } else if (settings["pipeline"][i]["type"].get<std::string>() ==
               "SynchronousProcessingUnit::cropFrame") {
      ptr = std::make_unique<CropFrame>();
    } else if (settings["pipeline"][i]["type"].get<std::string>() ==
               "SynchronousProcessingUnit::debug") {
      ptr = std::make_unique<Debug>();
    } else if (settings["pipeline"][i]["type"].get<std::string>() ==
               "SynchronousProcessingUnit::resizeFrame") {
      ptr = std::make_unique<ResizeFrame>();
    } else if (settings["pipeline"][i]["type"].get<std::string>() ==
               "SynchronousProcessingUnit::calculateChangeRate") {
      ptr = std::make_unique<CalculateChangeRate>();
    } else if (settings["pipeline"][i]["type"].get<std::string>() ==
               "SynchronousProcessingUnit::controlFps") {
      ptr = std::make_unique<ControlFps>();
    } else if (settings["pipeline"][i]["type"].get<std::string>() ==
               "AsynchronousProcessingUnit::videoWriter") {
      ptr = std::make_unique<VideoWriter>();
    } else if (settings["pipeline"][i]["type"].get<std::string>() ==
               "AsynchronousProcessingUnit::httpService") {
      ptr = std::make_unique<HttpService>();
    } else if (settings["pipeline"][i]["type"].get<std::string>() ==
               "AsynchronousProcessingUnit::zeroMqPublisher") {
      ptr = std::make_unique<ZmqPublisher>();
    } else {
      SPDLOG_WARN("Unrecognized pipeline unit: {}",
                  settings["pipeline"][i]["type"].get<std::string>());
      continue;
    }
    if (std::visit(
        overload{
            [&](const std::unique_ptr<ISynchronousProcessingUnit> &ptr_) {
              return ptr_->init(settings["pipeline"][i]);
            },
            [&](const std::unique_ptr<IAsynchronousProcessingUnit> &ptr_) {
              if (!ptr_->init(settings["pipeline"][i]))
                return false;
              ptr_->start();
              return true;
            },
        },
        ptr))
      processing_units.push_back(std::move(ptr));
    else {
      SPDLOG_WARN("not added");
    }
  }

  cv::cuda::GpuMat frame;
  cv::Ptr<cv::cudacodec::VideoReader> vr = nullptr;
  const auto video_feed = settings["device"]["uri"].get<std::string>();
  const auto expected_frame_width =
      settings["device"]["expectedFrameSize"]["width"].get<int>();
  const auto expected_frame_height =
      settings["device"]["expectedFrameSize"]["height"].get<int>();

  ProcessingMetaData meta_data;
  while (ev_flag == 0) {
    always_fill_in_frame(vr, expected_frame_height, expected_frame_width, frame,
                         meta_data);
    handle_video_capture(vr, meta_data, video_feed);

    for (size_t i = 0; i < processing_units.size(); ++i) {
      meta_data.processing_unit_idx = i;
      auto retval = std::visit(
          overload{
              [&](const std::unique_ptr<ISynchronousProcessingUnit> &ptr) {
                return ptr->process(frame, meta_data);
              },
              [&](const std::unique_ptr<IAsynchronousProcessingUnit> &ptr) {
                return ptr->enqueue(frame, meta_data);
              },
          },
          processing_units[i]);
      if (retval == failure_and_stop || retval == success_and_stop)
        break;
    }
  }

  vr.release();
  SPDLOG_INFO("[{}] thread quits gracefully", deviceName);
}

void DeviceManager::always_fill_in_frame(
    const cv::Ptr<cv::cudacodec::VideoReader> &vr,
    const int expected_frame_height, const int expected_frame_width,
    cv::cuda::GpuMat &frame, ProcessingUnit::ProcessingMetaData &meta_data) {
  auto captured_from_real_device = false;
  if (vr != nullptr) {
    try {
      if (!vr->nextFrame(frame)) {
        SPDLOG_ERROR("VideoReader->nextFrame(frame) returns false");
      } else if (frame.empty() ||
                 frame.size().height != expected_frame_height ||
                 frame.size().width != expected_frame_width) {
        SPDLOG_ERROR("VideoReader->nextFrame((frame) returns frame with "
                     "unexpected size. expect ({}x{}) vs actual ({}x{})",
                     expected_frame_width, expected_frame_height,
                     frame.size().width, frame.size().height);
      } else {
        captured_from_real_device = true;
      }
    } catch (const cv::Exception &e) {
      spdlog::error("VideoReader->nextFrame() failed: {}", e.what());
    }
  }

  if (!meta_data.captured_from_real_device) {
    // emulate an 30-fps video device lol
    this_thread::sleep_for(1000ms / 34);
    frame.create(expected_frame_height, expected_frame_width, CV_8UC3);
    frame.setTo(cv::Scalar(128, 128, 128));
  }
  meta_data.capture_timestamp_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  if (captured_from_real_device != meta_data.captured_from_real_device) {
    meta_data.capture_from_this_device_since_ms =
        meta_data.capture_timestamp_ms;
  }
  meta_data.captured_from_real_device = captured_from_real_device;
  ++meta_data.frame_seq_num;
}

void DeviceManager::handle_video_capture(
    cv::Ptr<cv::cudacodec::VideoReader> &vr,
    const ProcessingUnit::ProcessingMetaData &meta_data,
    const std::string &video_feed) {

  if (!meta_data.captured_from_real_device) [[unlikely]] {
    const auto device_down_for_sec =
        (std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
             .count() -
         meta_data.capture_from_this_device_since_ms) /
        1000;
    // at 30 fps, we at most wait for 10 min
    const auto wait_for_n_frame_before_retry =
        30 * std::max(device_down_for_sec, 60L * 10);
    if (meta_data.frame_seq_num % wait_for_n_frame_before_retry != 0 &&
        meta_data.frame_seq_num > 1 /* initial open */)
      return;

    if (meta_data.frame_seq_num > 1)
      SPDLOG_INFO("Waited for {} frames, about to retry "
                  "cv::cudacodec::createVideoReader({})",
                  wait_for_n_frame_before_retry, video_feed);
    auto params = cv::cudacodec::VideoReaderInitParams();
    // https://docs.opencv.org/4.9.0/dd/d7d/structcv_1_1cudacodec_1_1VideoReaderInitParams.html
    params.allowFrameDrop = true;
    try {
      vr = cv::cudacodec::createVideoReader(video_feed, {}, params);
      vr->set(cv::cudacodec::ColorFormat::BGR);
      SPDLOG_INFO("cv::cudacodec::createVideoReader({}) succeeded", video_feed);
    } catch (const cv::Exception &e) {
      spdlog::error("cudacodec::createVideoReader({}) failed: {}", video_feed,
                    e.what());
    }

  }
}
} // namespace CudaMotion
