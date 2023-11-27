#include "global_vars.h"
#include "ipc.h"
#include "pc_queue.h"

#include <spdlog/spdlog.h>

using namespace std;

template <>
PcQueue<cv::Mat, ipcQueuePayload, ipcQueuePayload>::PcQueue() : q(512) {}

template <>
bool PcQueue<cv::Mat, ipcQueuePayload, ipcQueuePayload>::try_enqueue(
    cv::Mat dispFrame) {
  return q.try_enqueue(dispFrame.clone());
}

template <>
void PcQueue<cv::Mat, ipcQueuePayload, ipcQueuePayload>::consumeCb(
    ipcQueuePayload pl) {
  pl.ipcInstance->sendDataCb(pl.snapshot);
}

template <>
void PcQueue<cv::Mat, ipcQueuePayload, ipcQueuePayload>::evConsume(
    PcQueue *This, ipcQueuePayload pl) {
  while (ev_flag == 0) {
    if (This->q.wait_dequeue_timed(pl.snapshot,
                                   std::chrono::milliseconds(100))) {
      This->consumeCb(pl);
    }
  }
}
