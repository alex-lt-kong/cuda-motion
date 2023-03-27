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
        int errNum;
        /* What happens if InternalThreadEntryFunc() throws an exception?
        Would you trigger undefined behavior as pthread_create() has no idea
        wtf is a C++ exception? The answer is no:
        Case I: if we catch it in the same thread, then it's fine, as always;
        Case II: if we don't catch it, then the thread will be. Unfortunately,
        the entire program will be terminated as well. But this is the same
        behavior as if we don't catch an exception in the main thread.
        */
        if ((errNum = pthread_create(&_thread, NULL,
            InternalThreadEntryFunc, this)) != 0) {
            throw runtime_error("pthread_create() failed: " +
                to_string(errNum) + " (" + strerror(errNum) + ")");
        }
    }

    void WaitForInternalEventLoopThreadToExit() {
        int errNum;
        if ((errNum = pthread_join(_thread, NULL)) != 0) {
            throw runtime_error("pthread_join() failed: " +
                to_string(errNum) + " (" + strerror(errNum) + ")");
        }
    }

    /**
     * @brief One should either WaitForInternalEventLoopThreadToExit() or
     * DetachInternalEventLoopThread()
    */
    void DetachInternalEventLoopThread() {        
        int errNum;
        if ((errNum = pthread_detach(_thread)) != 0) {
            throw runtime_error("pthread_detach() failed: " +
                to_string(errNum) + " (" + strerror(errNum) + ")");
        }
    }



protected:
    /** Implement this method in your subclass with the code you want your thread to run. */
    virtual void InternalThreadEntry() = 0;

private:
    static void * InternalThreadEntryFunc(void * This) {
        ((MyEventLoopThread *)This)->InternalThreadEntry();
        return NULL;
    }
    pthread_t _thread;
};

#endif
