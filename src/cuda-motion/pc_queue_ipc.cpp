// #include "global_vars.h"
#include "ipc.h"
#include "pc_queue.h"

#include <iostream>
#include <spdlog/spdlog.h>

using namespace std;

/*
template <>
PcQueue<ipcQueueElement, ipcDequeueContext>::PcQueue(volatile sig_atomic_t *)
    : q(512) {}*/

template <>
bool PcQueue<ipcQueueElement, ipcDequeueContext>::try_enqueue(
    ipcQueueElement eqpl) {
  return q.try_enqueue(eqpl);
}

template <>
void PcQueue<ipcQueueElement, ipcDequeueContext>::evConsume(
    PcQueue *This, ipcDequeueContext ctx) {
  while (!*(This->_ev_flag)) {
    if (This->q.wait_dequeue_timed(ctx.ele, chrono::milliseconds(100))) {
      ctx.ipcInstance->sendDataCb(ctx.ele);
    }
  }
}
