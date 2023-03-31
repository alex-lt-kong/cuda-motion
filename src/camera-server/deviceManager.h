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

#include "eventLoop.h"
#include "utils.h"

using namespace std;
using namespace cv;
using njson = nlohmann::json;

/* The extern keyword tells the compiler that please dont generate any
definition for it when compiling the source files that include the header.
Without extern, multiple object files that include this header file
will generate its own version of ev_flag, causing the "multiple definition
of `ev_flag';" error. By adding extern, we need to manually add the definition
of ev_flag in one .c/.cpp file. In this particular case, this is done
in main.cpp. */
extern volatile sig_atomic_t ev_flag;


#define PERMS (S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP)
#define SEM_INITIAL_VALUE 1


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
    uint32_t maxFramesPerVideo;
    uint32_t minFramesPerVideo;
    size_t frameQueueSize;

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
    void startOrKeepVideoRecording(VideoWriter& vwriter, int64_t& cooldown);
    void stopVideoRecording(VideoWriter& vwriter, uint32_t& videoFrameCount,
        int cooldown);
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
    void prepareDataForIpc(Mat& dispFrames);
    float getCurrentFps(int64_t msSinceEpoch);
};

#endif
