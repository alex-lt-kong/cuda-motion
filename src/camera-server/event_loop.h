#ifndef CS_EVENT_LOOP_H
#define CS_EVENT_LOOP_H

#include <stdexcept>
#include <string>
#include <thread>

// This multithreading model is inspired by:
// https://stackoverflow.com/questions/1151582/pthread-function-from-a-class
class MyEventLoopThread {
public:
  MyEventLoopThread() {}
  virtual ~MyEventLoopThread() {}

  void StartInternalEventLoopThread() {
    _thread = std::thread(&MyEventLoopThread::InternalThreadEntry, this);
  }

  void WaitForInternalEventLoopThreadToExit() {
    if (_thread.joinable()) {
      _thread.join();
    }
  }

  /**
   * @brief One should either WaitForInternalEventLoopThreadToExit() or
   * DetachInternalEventLoopThread()
   */
  void DetachInternalEventLoopThread() { _thread.detach(); }

protected:
  /** Implement this method in your subclass with the code you want your thread
   * to run. */
  virtual void InternalThreadEntry() = 0;

private:
  /*static void *InternalThreadEntryFunc(void *This) {
    ((MyEventLoopThread *)This)->InternalThreadEntry();
    return NULL;
  }*/
  std::thread _thread;
};

#endif /* CS_EVENT_LOOP_H */
