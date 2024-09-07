#ifndef CS_PC_QUEUE_H
#define CS_PC_QUEUE_H

#include <readerwriterqueue/readerwritercircularbuffer.h>

// #include <atomic>
// #include <iostream>
#include <signal.h>
#include <thread>

template <class T_QUEUE_ELEMENT, class T_DEQUEUE_CTX> class PcQueue {
private:
  moodycamel::BlockingReaderWriterCircularBuffer<T_QUEUE_ELEMENT> q;
  std::thread consumer;
  volatile sig_atomic_t *_ev_flag;
  static void evConsume(PcQueue *This, T_DEQUEUE_CTX);
  // void consumeCb(T_CB_PAYLOAD);

public:
  inline PcQueue(volatile sig_atomic_t *ev_flag, const size_t queue_size)
      : q(queue_size) {
    _ev_flag = ev_flag;
  }

  inline void start(T_DEQUEUE_CTX pl) {
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

  bool try_enqueue(T_QUEUE_ELEMENT);
};

#endif // CS_PC_QUEUE_H
