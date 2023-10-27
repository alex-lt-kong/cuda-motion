#ifndef CS_EVENT_LOOP_H
#define CS_EVENT_LOOP_H

#include <thread>
// This multithreading model is inspired by:
// https://stackoverflow.com/questions/1151582/pthread-function-from-a-class
class EventLoop {
public:
  EventLoop() {}
  virtual ~EventLoop() {}

  void StartEv() {
    evThread = std::thread(&EventLoop::InternalThreadEntry, this);
  }

  void JoinEv() {
    if (evThread.joinable()) {
      evThread.join();
    }
  }

  /**
   * @brief One should either WaitForInternalEventLoopThreadToExit() or
   * DetachInternalEventLoopThread()
   */
  void DetachEv() { evThread.detach(); }

protected:
  /** Implement this method in your subclass with the code you want your thread
   * to run. */
  virtual void InternalThreadEntry() = 0;

private:
  /*static void *InternalThreadEntryFunc(void *This) {
    ((MyEventLoopThread *)This)->InternalThreadEntry();
    return NULL;
  }*/
  std::thread evThread;
};

#endif /* CS_EVENT_LOOP_H */
