#include "device_manager.h"
#include "asynchronous_processing_units/http_service.h"
#include "asynchronous_processing_units/video_writer.h"
#include "asynchronous_processing_units/zeromq_publisher.h"
#include "global_vars.h"
#include "pipeline_executor.h"

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


void DeviceManager::InternalThreadEntry() {
  PipelineExecutor exe;
  if (!exe.init(settings["pipeline"])) {
    throw std::runtime_error("pipeline initialization failed");
    return;
  }


  cv::cuda::GpuMat frame;
  // cv::Ptr<cv::cudacodec::VideoReader> vr = nullptr;
  const auto video_feed = settings["device"]["uri"].get<std::string>();
  const auto expected_frame_width =
      settings["device"]["expectedFrameSize"]["width"].get<int>();
  const auto expected_frame_height =
      settings["device"]["expectedFrameSize"]["height"].get<int>();

  ProcessingUnit::PipelineContext ctx;
  ctx.capture_from_this_device_since_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  while (ev_flag == 0) {
    always_fill_in_frame(expected_frame_height, expected_frame_width, frame,
                         ctx);
    handle_video_capture(ctx, video_feed);
    exe.on_frame_ready(frame, ctx);
  }

  vr.release();
  SPDLOG_INFO("[{}] thread quits gracefully", deviceName);
}

void DeviceManager::always_fill_in_frame(
    const int expected_frame_height, const int expected_frame_width,
    cv::cuda::GpuMat &frame, ProcessingUnit::PipelineContext &ctx) {
  auto captured_from_real_device = false;

  try {
    std::lock_guard lock(mtx_vr);
    constexpr auto interval_by_frames = 90;
    if (vr == nullptr) {
      if (ctx.frame_seq_num % interval_by_frames == 0)
        SPDLOG_WARN("vr == nullptr, frame_seq_num: {} (this "
                     "message is throttled to once per {} frames)",
                     ctx.frame_seq_num, interval_by_frames);
    } else if (!vr->nextFrame(frame)) {
      if (ctx.frame_seq_num % interval_by_frames == 0)
        SPDLOG_ERROR("VideoReader->nextFrame(frame) returns false, "
                     "frame_seq_num: {} (this "
                     "message is throttled to once per {} frames)",
                     ctx.frame_seq_num, interval_by_frames);
    } else if (frame.empty() || frame.size().height != expected_frame_height ||
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

  if (!ctx.captured_from_real_device) {
    // emulate an 30-fps video device lol
    this_thread::sleep_for(1000ms / 34);
    frame.create(expected_frame_height, expected_frame_width, CV_8UC3);
    frame.setTo(cv::Scalar(128, 128, 128));
  }
  ctx.capture_timestamp_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count();
  if (captured_from_real_device != ctx.captured_from_real_device) {
    ctx.capture_from_this_device_since_ms =
        ctx.capture_timestamp_ms;
  }
  ctx.captured_from_real_device = captured_from_real_device;
  ++ctx.frame_seq_num;
}

void DeviceManager::handle_video_capture(
    const ProcessingUnit::PipelineContext &ctx, const std::string &video_feed) {

  auto register_delayed_vc_open_retry = [this](const ProcessingUnit::
                                                   PipelineContext &ctx,
                                               const std::string &video_feed) {
    {
      std::lock_guard lock(mtx_vr);
      if (delayed_vc_open_retry_registered)
        return;
      delayed_vc_open_retry_registered = true;
    }

    const auto device_down_for_sec =
        (std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
             .count() -
         ctx.capture_from_this_device_since_ms) /
        1000;
    const auto delay_sec_before_attempt =
        std::min(std::max(device_down_for_sec, 2L), 60L * 10);
    SPDLOG_WARN("captured_from_real_device: {}, device_down_for_sec: {}, "
                "delay_sec_before_attempt: {}",
                ctx.captured_from_real_device, device_down_for_sec,
                delay_sec_before_attempt);
    std::thread t(
        [this](const long delay_sec_before_attempt, std::string video_feed) {
          const auto delay_sec_before_attempt_ = delay_sec_before_attempt;
          SPDLOG_INFO(
              "timer started, delay_sec_before_attempt ({}) and then invoke "
              "cv::cudacodec::createVideoReader({})",
              delay_sec_before_attempt_, video_feed);
          std::this_thread::sleep_for(
              std::chrono::seconds(delay_sec_before_attempt_));
          {
            std::lock_guard lock(mtx_vr);
            SPDLOG_INFO(
                "delay_sec_before_attempt ({}) reached, about to invoke "
                "cv::cudacodec::createVideoReader({})",
                delay_sec_before_attempt_, video_feed);
            auto params = cv::cudacodec::VideoReaderInitParams();
            // https://docs.opencv.org/4.9.0/dd/d7d/structcv_1_1cudacodec_1_1VideoReaderInitParams.html
            params.allowFrameDrop = true;
            try {
              vr = cv::cudacodec::createVideoReader(video_feed, {}, params);
              vr->set(cv::cudacodec::ColorFormat::BGR);
              SPDLOG_INFO("cv::cudacodec::createVideoReader({}) succeeded",
                          video_feed);
            } catch (const cv::Exception &e) {
              spdlog::error("cudacodec::createVideoReader({}) failed: {}",
                            video_feed, e.what());
            }
            // need to wait so that the event loop will not register a timer too soon
            std::this_thread::sleep_for(5000ms);
            delayed_vc_open_retry_registered = false;
          }
        },
        delay_sec_before_attempt, video_feed);
    t.detach();
  };

  if (!ctx.captured_from_real_device) [[unlikely]] {
    register_delayed_vc_open_retry(ctx, video_feed);
  }
}
} // namespace CudaMotion
