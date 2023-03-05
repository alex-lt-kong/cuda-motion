#include <string>
#include <queue>
#include <sys/time.h>
#include <signal.h>
#include <pthread.h>

#include <nlohmann/json.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using json = nlohmann::json;


// This multithreading model is inspired by:
// https://stackoverflow.com/questions/1151582/pthread-function-from-a-class
class MyEventLoopThread {
public:
    MyEventLoopThread() {}
    virtual ~MyEventLoopThread() {}

    /** Returns true if the thread was successfully started, false if there was an error starting the thread */
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


class deviceManager : public MyEventLoopThread {

public:
    deviceManager();
    ~deviceManager();
    void setParameters(json settings);
    void getLiveImage(vector<uint8_t>& pl);

protected:
    void InternalThreadEntry();


private:
    pthread_mutex_t mutexLiveImage;
    vector<uint8_t> encodedJpgImage;
    bool enableContoursDrawing = false;
    double fontScale = 1;
    double rateOfChangeUpper = 0;
    double rateOfChangeLower = 0;
    double pixelLevelThreshold = 0;
    int snapshotFrameInterval = 1;
    int frameRotation = -1;
    int framePreferredWidth = -1;
    int framePreferredHeight = -1;
    int framePreferredFps = -1;
    int frameFpsUpperCap = 1;
    int framesAfterTrigger = 0;
    long long int maxFramesPerVideo = 1;
    int diffFrameInterval = 1;
    int frameIntervalInMs = 24;
    string deviceUri;
    string deviceName;   
    string ffmpegCommand;
    string snapshotPath;
    string eventOnVideoStarts;
    string eventOnVideoEnds;
    queue<long long int> frameTimestamps;

    volatile sig_atomic_t* done;
    
    bool skipThisFrame();
    string convertToString(char* a, int size);
    string getCurrentTimestamp();
    void rateOfChangeInRange(FILE** ffmpegPipe, int* cooldown, string* timestampOnVideoStarts);
    void coolDownReachedZero(FILE** ffmpegPipe, uint32_t* videoFrameCount, string* timestampOnVideoStarts);
    void overlayDatetime(Mat frame);
    void overlayDeviceName(Mat frame);
    void overlayContours(Mat dispFrame, Mat diffFrame);
    void overlayChangeRate(Mat frame, float changeRate, int cooldown, long long int videoFrameCount);
    float getFrameChanges(Mat prevFrame, Mat currFrame, Mat* diffFrame);
};
