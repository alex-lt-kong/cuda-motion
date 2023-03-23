#ifndef EVENT_LOOP_H
#define EVENT_LOOP_H

#include <string>

using namespace std;

// This multithreading model is inspired by:
// https://stackoverflow.com/questions/1151582/pthread-function-from-a-class
class MyEventLoopThread {
public:
    MyEventLoopThread() {}
    virtual ~MyEventLoopThread() {}

    void StartInternalEventLoopThread() {
        if (_internalThreadShouldQuit == false) {
            throw runtime_error("StartInternalEventLoopThread() is called "
                "when the internal thread is still running");
        }
        _internalThreadShouldQuit = false;
        if (pthread_create(&_thread, NULL,
            InternalThreadEntryFunc, this) != 0) {
            throw runtime_error("pthread_create() failed, errno: " +
                to_string(errno));
        }
    }

    /**
     * @brief set the _internalThreadShouldQuit. Users should check this
     * signal periodically and quit the event loop timely based on the signal.
    */
    void StopInternalEventLoopThread() {
        _internalThreadShouldQuit = true;
    }

    void WaitForInternalEventLoopThreadToExit() {
        pthread_join(_thread, NULL);
    }

    /**
     * @brief One should either WaitForInternalEventLoopThreadToExit() or
     * DetachInternalEventLoopThread()
    */
    void DetachInternalEventLoopThread() {
        if (pthread_detach(_thread) == 0) {
            throw runtime_error("failed to pthread_detach() a thread, errno: " +
                to_string(errno));
        }
    }



protected:
    /** Implement this method in your subclass with the code you want your thread to run. */
    virtual void InternalThreadEntry() = 0;
    bool _internalThreadShouldQuit = true;

private:
    static void * InternalThreadEntryFunc(void * This) {
        ((MyEventLoopThread *)This)->InternalThreadEntry();
        return NULL;
    }
    pthread_t _thread = 0;
};

#endif
