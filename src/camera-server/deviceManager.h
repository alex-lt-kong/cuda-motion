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


// This multithreading model is from: https://stackoverflow.com/questions/1151582/pthread-function-from-a-class
class MyThreadClass {
public:
    MyThreadClass() {}
    virtual ~MyThreadClass() {}

    /** Returns true if the thread was successfully started, false if there was an error starting the thread */
    void StartInternalThread() {
        if (pthread_create(&_thread, NULL,
            InternalThreadEntryFunc, this) != 0) {
            throw runtime_error("pthread_create() failed, errno: " +
                to_string(errno));
        }
    }

    void WaitForInternalThreadToExit() {
        pthread_join(_thread, NULL);
    }

protected:
    /** Implement this method in your subclass with the code you want your thread to run. */
    virtual void InternalThreadEntry() = 0;

private:
    static void * InternalThreadEntryFunc(void * This) {((MyThreadClass *)This)->InternalThreadEntry(); return NULL;}
    pthread_t _thread = 0;
};


class deviceManager : public MyThreadClass {

public:
    deviceManager();
    ~deviceManager();
    bool captureImage(string imageSaveTo);
    bool setParameters(json settings, volatile sig_atomic_t* done);
    void getLiveImage(vector<uint8_t>& pl);
    size_t totalCount;

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
    string deviceUri = "";
    string deviceName = "";   
    string ffmpegCommand = "";
    string snapshotPath = "";
    string eventOnVideoStarts = "";
    string eventOnVideoEnds = "";
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
