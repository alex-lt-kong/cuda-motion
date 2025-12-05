#pragma once

#include "../entities/processing_metadata.h"
#include "../entities/synchronous_processing_result.h"

#include <nlohmann/json.hpp>
#include <opencv2/cudawarping.hpp>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <spdlog/spdlog.h>
#include <thread>

using njson = nlohmann::json;

namespace CudaMotion::ProcessingUnit {

struct AsyncPayload {
  cv::cuda::GpuMat frame;
  PipelineContext meta_data;
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
      std::lock_guard<std::mutex> lock(m_queue_mutex);
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
  virtual void stop() {
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
                       PipelineContext &meta_data) = 0;

private:
  /**
   * @brief The internal thread loop.
   * Pops items from the queue and delegates to process_frame().
   */
  void dequeue_loop() {
    while (m_running.load() || !is_queue_empty()) {
      AsyncPayload payload;

      {
        std::unique_lock<std::mutex> lock(m_queue_mutex);

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
        if (const auto queue_size = m_processing_queue.size();
            queue_size > 30) {
          spdlog::warn("Processing queue size too large ({})", queue_size);
        }
      }
      on_frame_ready(payload.frame, payload.meta_data);
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

} // namespace CudaMotion::ProcessingUnit