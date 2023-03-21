#ifndef DEVICE_MANAGER_H
#define DEVICE_MANAGER_H

#include <linux/stat.h>
#include <string>
#include <pthread.h>
#include <queue>
#include <semaphore.h>
#include <sys/time.h>
#include <signal.h>


#include <nlohmann/json.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "utils.h"

using namespace std;
using namespace cv;
using njson = nlohmann::json;

#define PERMS (S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP)
#define SEM_INITIAL_VALUE 1

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
    deviceManager(const size_t deviceIndex, const njson& defaultConf, njson& overrideConf);
    ~deviceManager();
    void setParameters(const size_t deviceIndex, const njson& defaultConf,
        njson& overrideConf);
    void getLiveImage(vector<uint8_t>& pl);
    string getDeviceName() { return this->deviceName; }

protected:
    void InternalThreadEntry();


private:
    pthread_mutex_t mutexLiveImage;
    vector<uint8_t> encodedJpgImage;
    njson conf;
    size_t deviceIndex = 0;
    string deviceName;

    // frame variables
    bool textOverlayEnabled;
    double textOverlayFontSacle;
    float throttleFpsIfHigherThan;
    int frameRotation;

    // motionDetection variables
    enum MotionDetectionMode motionDetectionMode;
    double frameDiffPercentageUpperLimit = 0;
    double frameDiffPercentageLowerLimit = 0;
    double pixelDiffAbsThreshold = 0;    
    uint64_t diffEveryNthFrame = 1;
    bool drawContours;

    // videoRecording variables   
    bool encoderUseExternal; 
    string pipeRawVideoTo;
    uint32_t maxFramesPerVideo;
    uint32_t minFramesPerVideo;
    size_t precaptureFrames;

    // snapshot variables
    int snapshotFrameInterval;
    bool snapshotIpcFileEnabled;
    bool snapshotIpcHttpEnabled;
    string snapshotIpcFilePath;
    bool snapshotHttpFileEnabled;
    bool snapshotIpcSharedMemEnabled;
    int shmFd;
    size_t sharedMemSize;
    string sharedMemName;
    string semaphoreName;
    void* memPtr;
    sem_t* semPtr;


    string timestampOnVideoStarts;
    string timestampOnDeviceOffline;
    queue<int64_t> frameTimestamps;

    volatile sig_atomic_t* done;
    
    void updateVideoCooldownAndVideoFrameCount(int64_t& cooldown,
        uint32_t& videoFrameCount);
    bool shouldFrameBeThrottled();
    string evaluateStaticVariables(basic_string<char> originalString);
    string evaluateVideoSpecficVariables(basic_string<char> originalString);
    string convertToString(char* a, int size);
    string getCurrentTimestamp();
    void startOrKeepVideoRecording(FILE*& ffmpegPipe, VideoWriter& vwriter,
        int64_t& cooldown);
    void stopVideoRecording(FILE*& ffmpegPipe, VideoWriter& vwriter,
        uint32_t& videoFrameCount, int cooldown);
    void overlayDatetime(Mat& frame);
    void overlayDeviceName(Mat& frame);
    void overlayContours(Mat& dispFrame, Mat& diffFrame);
    void overlayStats(Mat& frame, float changeRate, int cooldown,
        long long int videoFrameCount);
    float getFrameChanges(Mat& prevFrame, Mat& currFrame, Mat* diffFrame);
    void generateBlankFrameAt1Fps(Mat& currFrame, const Size& actualFrameSize);
    void deviceIsOffline(Mat& currFrame, const Size& actualFrameSize,
        bool& isShowingBlankFrame);
    void deviceIsBackOnline(size_t& openRetryDelay, bool& isShowingBlankFrame);
    void initializeDevice(VideoCapture& cap, bool& result,
        const Size& actualFrameSize);
    static void asyncExecCallback(void* This, string stdout, string stderr,
        int rc);
    void prepareDataForIpc(queue<cv::Mat>& dispFrames);
    float getCurrentFps(int64_t msSinceEpoch);
};

#endif
