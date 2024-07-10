#include "device_manager.h"
#include "pc_queue.h"
#include <spdlog/spdlog.h>

using namespace std;

template <>
PcQueue<cv::cuda::GpuMat, struct videoWritingInfo,
        struct videoWritingPayload>::PcQueue()
    : q(512) {}

template <>
void PcQueue<cv::cuda::GpuMat, struct videoWritingInfo,
             struct videoWritingPayload>::consumeCb(struct videoWritingPayload
                                                        pl) {
  if (pl.vw != nullptr)
    pl.vw->write(pl.m);
}

template <>
void PcQueue<cv::cuda::GpuMat, struct videoWritingInfo,
             struct videoWritingPayload>::
    evConsume(PcQueue<cv::cuda::GpuMat, struct videoWritingInfo,
                      struct videoWritingPayload> *This,
              struct videoWritingInfo inf) {
  Ptr<cudacodec::VideoWriter> vw = nullptr;
  try {
    vw = cudacodec::createVideoWriter(
        inf.evaluatedVideoPath, Size(inf.outputWidth, inf.outputHeight),
        cudacodec::Codec::H264, inf.fps, cudacodec::ColorFormat::BGR);
  } catch (const cv::Exception &e) {
    spdlog::error(
        "cudacodec::createVideoWriter() failed to initialize for file {}: {}",
        inf.evaluatedVideoPath, e.what());
  }
  struct videoWritingPayload pl;
  pl.vw = vw;
  while (ev_flag == 0 && inf.videoWriting) {
    if (This->q.wait_dequeue_timed(pl.m, std::chrono::milliseconds(500))) {
      This->consumeCb(pl);
    }
  }
  if (vw != nullptr)
    vw->release();
}

template <>
bool PcQueue<cv::cuda::GpuMat, struct videoWritingInfo,
             struct videoWritingPayload>::try_enqueue(const cv::cuda::GpuMat
                                                          dispFrame) {
  // cv::cuda::GpuMat operates with an internal reference-counter, so we need to
  // clone() to increase the counter Another point is that try_enqueue() does
  // std::move() internally, how does it interplay with cv::cuda::GpuMat's
  // ref-counting model? Not 100% clear to me...
  return q.try_enqueue(dispFrame);
}
