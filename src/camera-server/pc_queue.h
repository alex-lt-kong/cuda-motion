#ifndef PC_QUEUE_H
#define PC_QUEUE_H

#include <readerwriterqueue/readerwritercircularbuffer.h>

#include <atomic>
#include <iostream>
#include <thread>

template <class T_ENQUEUE_PAYLOAD, class T_CONSUME_PAYLOAD, class T_CB_PAYLOAD>
class PcQueue {
private:
  moodycamel::BlockingReaderWriterCircularBuffer<T_ENQUEUE_PAYLOAD> q;
  std::thread consumer;
  static void evConsume(PcQueue *This, T_CONSUME_PAYLOAD);
  void consumeCb(T_CB_PAYLOAD);

public:
  PcQueue();

  inline void start(T_CONSUME_PAYLOAD pl) {
    consumer = std::thread(&evConsume, this, pl);
  }

  inline void wait() {
    if (consumer.joinable()) {
      consumer.join();
    }
  }

  inline ~PcQueue() {
    // wait() is not thread-safe, can't call it here
  }

  bool try_enqueue(T_ENQUEUE_PAYLOAD);
};

#endif // RS_PC_QUEUE_H
