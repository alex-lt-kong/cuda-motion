#include "device_manager.h"
#include "pc_queue.h"

using namespace std;

template <>
PcQueue<cv::Mat, struct videoWritingInfo, struct videoWritingPayload>::PcQueue()
    : q(512) {}

template <>
void PcQueue<cv::Mat, struct videoWritingInfo, struct videoWritingPayload>::
    consumeCb(struct videoWritingPayload pl) {
  pl.vw.write(pl.m);
}

template <>
void PcQueue<cv::Mat, struct videoWritingInfo, struct videoWritingPayload>::
    evConsume(PcQueue<cv::Mat, struct videoWritingInfo,
                      struct videoWritingPayload> *This,
              struct videoWritingInfo inf) {
  const int codec = VideoWriter::fourcc(inf.fourcc[0], inf.fourcc[1],
                                        inf.fourcc[2], inf.fourcc[3]);
  /* For VideoWriter, we have to use FFmpeg as we compiled FFmpeg with
  Nvidia GPU*/
  auto vw = VideoWriter(
      inf.evaluatedVideoPath, cv::CAP_FFMPEG, codec, inf.fps,
      Size(inf.outputWidth, inf.outputHeight),
      {VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY});
  struct videoWritingPayload pl;
  pl.vw = vw;
  while (ev_flag == 0 && inf.videoWriting) {
    if (This->q.wait_dequeue_timed(pl.m, std::chrono::milliseconds(500))) {
      This->consumeCb(pl);
    }
  }
  vw.release();
}

template <>
bool PcQueue<cv::Mat, struct videoWritingInfo,
             struct videoWritingPayload>::try_enqueue(const cv::Mat dispFrame) {
  // cv::Mat operates with an internal reference-counter, so we need to clone()
  // to increase the counter
  // Another point is that try_enqueue() does std::move() internally, how does
  // it interplay with cv::Mat's ref-counting model? Not 100% clear to me...
  return q.try_enqueue(dispFrame);
}
