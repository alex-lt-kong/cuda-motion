#pragma once

#include "../entities/processing_context.h"
#include "../entities/synchronous_processing_result.h"
#include "../entities/video_recording_state.h"
#include "../utils/matrix_sender.h"
#include "../utils/ram_video_buffer.h"

#include <nlohmann/json.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudacodec.hpp>
#include <spdlog/spdlog.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

namespace MatrixPipeline {
namespace Utils {

class NvJpegEncoder;
} // namespace Utils
class PipelineExecutor;
} // namespace MatrixPipeline
using njson = nlohmann::json;

namespace MatrixPipeline::ProcessingUnit {

struct AsyncPayload {
  cv::cuda::GpuMat frame;
  PipelineContext ctx;
};

class IAsynchronousProcessingUnit {
public:
  virtual ~IAsynchronousProcessingUnit() {
    stop(); // Ensure thread is joined on destruction
  }

  virtual bool init(const njson &config) = 0;

  /**
   * @brief Pushes a frame into the processing queue.
   * * IMPORTANT: This performs a DEEP COPY (clone) of the GPU frame.
   * This ensures the worker thread owns independent data, allowing the caller
   * to reuse or destroy the original frame immediately after this call returns.
   * * Note: This involves GPU memory allocation, so it has a higher cost than a
   * shallow copy.
   */
  SynchronousProcessingResult enqueue(const cv::cuda::GpuMat &frame,
                                      const PipelineContext &meta_data) {
    const auto frame_deep_copy = frame.clone();
    {
      std::lock_guard lock(m_queue_mutex);
      m_processing_queue.push(AsyncPayload{frame_deep_copy, meta_data});
    }

    // 3. Wake up the worker thread
    m_cv.notify_one();
    return success_and_continue;
  }

  /**
   * @brief Starts the internal worker thread.
   */
  void start() {
    if (m_running.load()) {
      return; // Already running
    }
    m_running.store(true);
    m_worker_thread =
        std::thread(&IAsynchronousProcessingUnit::dequeue_loop, this);
  }

  /**
   * @brief Stops the worker thread and waits for it to finish.
   */
  void stop() {
    if (!m_running.load()) {
      return;
    }

    m_running.store(false);
    m_cv.notify_all(); // Wake up thread if it is sleeping

    if (m_worker_thread.joinable()) {
      m_worker_thread.join();
    }
  }

protected:
  /**
   * @brief The actual processing logic to be implemented by concrete classes.
   * This is called automatically by the internal thread when data is dequeued.
   */
  virtual void on_frame_ready(cv::cuda::GpuMat &frame,
                              PipelineContext &ctx) = 0;

private:
  /**
   * @brief The internal thread loop.
   * Pops items from the queue and delegates to process_frame().
   */
  void dequeue_loop() {
    while (m_running.load() || !is_queue_empty()) {
      AsyncPayload payload;

      {
        std::unique_lock lock(m_queue_mutex);

        // Wait until the queue has data OR we are asked to stop
        m_cv.wait(lock, [this] {
          return !m_processing_queue.empty() || !m_running.load();
        });

        if (m_processing_queue.empty()) {
          if (!m_running.load())
            break;  // Exit condition
          continue; // Spurious wake-up check
        }

        payload = m_processing_queue.front();
        m_processing_queue.pop();
        auto queue_size = m_processing_queue.size();
        if (constexpr auto warning_queue_size = 10;
            queue_size > warning_queue_size) {
          constexpr auto critical_queue_size = 30;
          SPDLOG_WARN("queue_size ({}) is above warning_queue_size ({})",
                      queue_size, warning_queue_size);
          while (queue_size > critical_queue_size) {
            auto [frame, ctx] = m_processing_queue.front();
            m_processing_queue.pop();
            queue_size = m_processing_queue.size();
            SPDLOG_ERROR("queue_size ({}) is above critical_queue_size ({}), "
                         "frame {} discarded to avoid OOM",
                         queue_size, critical_queue_size, ctx.frame_seq_num);
          }
        }
      }
      on_frame_ready(payload.frame, payload.ctx);
    }
  }

  bool is_queue_empty() {
    std::lock_guard<std::mutex> lock(m_queue_mutex);
    return m_processing_queue.empty();
  }

  std::queue<AsyncPayload> m_processing_queue;
  std::mutex m_queue_mutex;
  std::condition_variable m_cv;
  std::atomic<bool> m_running{false};
  std::thread m_worker_thread;
};

class AsynchronousProcessingUnit final : public IAsynchronousProcessingUnit {
  std::unique_ptr<PipelineExecutor> m_exe{nullptr};

public:
  bool init(const njson &config) override;
  void on_frame_ready(cv::cuda::GpuMat &frame, PipelineContext &ctx) override;
};

class MatrixNotifier final : public IAsynchronousProcessingUnit {
  std::unique_ptr<Utils::NvJpegEncoder> m_gpu_encoder{nullptr};
  std::unique_ptr<Utils::MatrixSender> m_sender{nullptr};
  std::string m_matrix_homeserver;
  std::string m_matrix_room_id;
  std::string m_matrix_access_token;
  int m_notification_interval_frame{300};
  bool m_is_send_image_enabled{true};
  bool m_is_send_video_enabled{true};
  size_t m_video_max_length_in_frame{30 * 10};
  size_t m_video_max_length_without_people_in_frame{30 * 10};

  size_t m_current_video_length_in_frame{0};
  size_t m_current_video_length_without_people_in_frame{0};
  float m_max_roi_value{0.0f};
  cv::cuda::GpuMat m_max_roi_value_frame{-1};
  const double m_target_fps{25.0};
  cv::Ptr<cv::cudacodec::VideoWriter> m_writer{nullptr};
  std::unique_ptr<Utils::RamVideoBuffer> m_ram_buf{nullptr};
  Utils::VideoRecordingState m_state{Utils::IDLE};

  static bool check_if_people_detected(const PipelineContext &ctx);

  void handle_image(const cv::cuda::GpuMat &frame,
                    [[maybe_unused]] const PipelineContext &ctx,
                    const bool is_people_detected) const;
  void handle_video(const cv::cuda::GpuMat &frame,
                    [[maybe_unused]] const PipelineContext &ctx,
                    const bool is_people_detected);

  static float calculate_roi_value(const YoloContext &yolo);

public:
  bool init(const njson &config) override;
  void on_frame_ready(cv::cuda::GpuMat &frame, PipelineContext &ctx) override;
};

} // namespace MatrixPipeline::ProcessingUnit