#include "device_manager.h"
#include "pc_queue.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudacodec.hpp>
#include <spdlog/spdlog.h>

using namespace std;
using namespace cv::cuda;
using namespace cv;
typedef struct videoWritingContext vwc;

/*
template <>
PcQueue<GpuMat, vwc>::PcQueue(volatile sig_atomic_t *ev_flag) : q(512) {}*/

template <>
void PcQueue<GpuMat, vwc>::evConsume(PcQueue<GpuMat, vwc> *This, vwc inf) {
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

  while (!*(This->_ev_flag) && inf.videoWriting) {
    GpuMat gm;
    if (This->q.wait_dequeue_timed(gm, std::chrono::milliseconds(500))) {
      if (vw != nullptr)
        vw->write(gm);
    }
  }
  if (vw != nullptr)
    vw->release();
}

template <> bool PcQueue<GpuMat, vwc>::try_enqueue(const GpuMat dispFrame) {
  // cv::cuda::GpuMat operates with an internal reference-counter, so we need to
  // clone() to increase the counter Another point is that try_enqueue() does
  // std::move() internally, how does it interplay with cv::cuda::GpuMat's
  // ref-counting model? Not 100% clear to me...
  return q.try_enqueue(dispFrame);
}
