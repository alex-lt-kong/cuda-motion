#include <arpa/inet.h>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <regex>
#include <sstream>
#include <sys/socket.h>

#include <spdlog/spdlog.h>

#include "deviceManager.h"

using json = nlohmann::json;
using namespace std;
using namespace cv;

using sysclock_t = std::chrono::system_clock;

string deviceManager::getCurrentTimestamp()
{
    std::time_t now = sysclock_t::to_time_t(sysclock_t::now());
    //"19700101_000000"
    char buf[16] = { 0 };
    std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", localtime(&now));
    // https://www.cplusplus.com/reference/ctime/strftime/
    return std::string(buf);
}


string deviceManager::convertToString(char* a, int size)
{
    int i;
    string s = "";
    for (i = 0; i < size; i++) {
        s = s + a[i];
    }
    return s;
}

bool deviceManager::setParameters(json settings, volatile sig_atomic_t* done) {
    this->deviceUri = settings["uri"];
    this->deviceName = settings["name"];
    this->frameRotation = settings["frame"]["rotation"];
    this->framePreferredWidth = settings["frame"]["preferredWidth"];
    this->framePreferredHeight = settings["frame"]["preferredHeight"];
    this->framePreferredFps = settings["frame"]["preferredFps"];
    this->frameFpsUpperCap = settings["frame"]["FpsUpperCap"];
    this->fontScale = settings["frame"]["overlayTextFontScale"];
    this->enableContoursDrawing = settings["frame"]["enableContoursDrawing"];
    this->snapshotPath = settings["snapshot"]["path"];
    this->snapshotFrameInterval = settings["snapshot"]["frameInterval"];  
    this->eventOnVideoStarts = settings["events"]["onVideoStarts"];
    this->eventOnVideoEnds = settings["events"]["onVideoEnds"];
    this->ffmpegCommand = settings["ffmpegCommand"];
    this->rateOfChangeLower = settings["motionDetection"]["frameLevelRateOfChangeLowerLimit"];
    this->rateOfChangeUpper = settings["motionDetection"]["frameLevelRateOfChangeUpperLimit"];
    this->pixelLevelThreshold = settings["motionDetection"]["pixelLevelDiffThreshold"];
    this->diffFrameInterval = settings["motionDetection"]["diffFrameInterval"];
    this->framesAfterTrigger = settings["video"]["framesAfterTrigger"];
    this->maxFramesPerVideo = settings["video"]["maxFramesPerVideo"];
    
    this->frameIntervalInMs = 1000 * (1.0 / this->frameFpsUpperCap);

    this->done = done;
    return true;
}

deviceManager::deviceManager() {
    if (pthread_mutex_init(&mutexLiveImage, NULL) != 0) {
        throw runtime_error("pthread_mutex_init() failed, errno: " +
            to_string(errno));
    }
}

deviceManager::~deviceManager() {
    pthread_mutex_destroy(&mutexLiveImage);
}

void deviceManager::overlayDatetime(Mat frame) {
    time_t now;
    time(&now);
    char buf[sizeof "1970-01-01 00:00:00"];
    strftime(buf, sizeof buf, "%F %T", localtime(&now));
    cv::Size textSize = getTextSize(buf, FONT_HERSHEY_DUPLEX, this->fontScale, 8 * this->fontScale, nullptr);
    putText(frame, buf, Point(5, textSize.height * 1.05), FONT_HERSHEY_DUPLEX, this->fontScale, Scalar(0,  0,  0  ), 8 * this->fontScale, LINE_AA, false);
    putText(frame, buf, Point(5, textSize.height * 1.05), FONT_HERSHEY_DUPLEX, this->fontScale, Scalar(255,255,255), 2 * this->fontScale, LINE_AA, false);
    /*
    void cv::putText 	(InputOutputArray  	img,
                        const String &  	text,
                        Point  	org,
                        int  	fontFace,
                        double  	fontScale,
                        Scalar  	color,
                        int  	thickness = 1,
                        int  	lineType = LINE_8,
                        bool  	bottomLeftOrigin = false 
        ) 	
    */
}


float deviceManager::getFrameChanges(Mat prevFrame, Mat currFrame, Mat* diffFrame) {
    if (prevFrame.empty() || currFrame.empty()) { return -1; }
    if (prevFrame.cols != currFrame.cols || prevFrame.rows != currFrame.rows) { return -1; }
    if (prevFrame.cols == 0 || prevFrame.rows == 0) { return -1; }
    
    absdiff(prevFrame, currFrame, *diffFrame);
    cvtColor(*diffFrame, *diffFrame, COLOR_BGR2GRAY);
    threshold(*diffFrame, *diffFrame, this->pixelLevelThreshold, 255, THRESH_BINARY);
    int nonZeroPixels = countNonZero(*diffFrame);
    return 100.0 * nonZeroPixels / (diffFrame->rows * diffFrame->cols);
}

void deviceManager::overlayDeviceName(Mat frame) {

    cv::Size textSize = getTextSize(this->deviceName, FONT_HERSHEY_DUPLEX, this->fontScale, 8 * this->fontScale, nullptr);
    putText(frame, this->deviceName, Point(frame.cols - textSize.width * 1.05, frame.rows - 5), 
            FONT_HERSHEY_DUPLEX, this->fontScale, Scalar(0,  0,  0  ), 8 * this->fontScale, LINE_AA, false);
    putText(frame, this->deviceName, Point(frame.cols - textSize.width * 1.05, frame.rows - 5),
            FONT_HERSHEY_DUPLEX, this->fontScale, Scalar(255,255,255), 2 * this->fontScale, LINE_AA, false);
}

void deviceManager::overlayChangeRate(Mat frame, float changeRate, int cooldown, long long int videoFrameCount) {
    int value = changeRate * 100;
    stringstream ssChangeRate;
    ssChangeRate << fixed << setprecision(2) << changeRate;
    putText(frame, ssChangeRate.str() + "% (" + to_string(cooldown) + ", " + to_string(this->maxFramesPerVideo - videoFrameCount) + ")", 
            Point(5, frame.rows-5), FONT_HERSHEY_DUPLEX, this->fontScale, Scalar(0,   0,   0  ), 8 * this->fontScale, LINE_AA, false);
    putText(frame, ssChangeRate.str() + "% (" + to_string(cooldown) + ", " + to_string(this->maxFramesPerVideo - videoFrameCount) + ")",
            Point(5, frame.rows-5), FONT_HERSHEY_DUPLEX, this->fontScale, Scalar(255, 255, 255), 2 * this->fontScale, LINE_AA, false);
}

void deviceManager::overlayContours(Mat dispFrame, Mat diffFrame) {
    if (diffFrame.empty()) return;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(diffFrame, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE );
    int idx = 0;
    for( ; idx >= 0; idx = hierarchy[idx][0] ) {
        drawContours(dispFrame, contours, idx, Scalar(255, 255, 255), 0.25, 8, hierarchy);
    }
}

bool deviceManager::skipThisFrame() {
    int sampleMsLowerLimit = 1000;
    int sampleMsUpperLimit = 60 * 1000;
    int currMsSinceEpoch = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
    if (this->frameTimestamps.size() <= 1) { 
        this->frameTimestamps.push(currMsSinceEpoch); 
        return false;
    }
    
    float fps = 1000.0 * this->frameTimestamps.size() / (1 + currMsSinceEpoch - this->frameTimestamps.front());
    if (currMsSinceEpoch - this->frameTimestamps.front() > sampleMsUpperLimit) {
        this->frameTimestamps.pop();
    }
    if (fps > this->frameFpsUpperCap) { return true; }
    this->frameTimestamps.push(currMsSinceEpoch);  
    return false;
}

void deviceManager::coolDownReachedZero(
  FILE** ffmpegPipe, uint32_t* videoFrameCount, string* timestampOnVideoStarts
) {
    if (*ffmpegPipe != nullptr) { // No, you cannot pclose() a nullptr
        pclose(*ffmpegPipe); 
        *ffmpegPipe = nullptr; 
        
        if (this->eventOnVideoEnds.length() > 0) {
        spdlog::info("[{}] video recording ends", this->deviceName);
        string commandOnVideoEnds = regex_replace(
            this->eventOnVideoEnds, regex("\\{\\{timestamp\\}\\}"), *timestampOnVideoStarts
        );
        system((commandOnVideoEnds + " &").c_str());

        spdlog::info("[{}] onVideoEnds triggered, command [{}] executed", this->deviceName, commandOnVideoEnds);
        } else {
        spdlog::info("[{}] onVideoEnds, no command to execute", this->deviceName);
        }
    }
    *videoFrameCount = 0;
}


void deviceManager::rateOfChangeInRange(
  FILE** ffmpegPipe, int* cooldown, string* timestampOnVideoStarts
) {
    *cooldown = this->framesAfterTrigger;
    if (*ffmpegPipe == nullptr) {
        string command = this->ffmpegCommand;
        *timestampOnVideoStarts = this->getCurrentTimestamp();
        command = regex_replace(
        command, regex("\\{\\{timestamp\\}\\}"), *timestampOnVideoStarts
        );
        *ffmpegPipe = popen((command).c_str(), "w");
        //setvbuf(*ffmpegPipe, &(this->buff), _IOFBF, this->buff_size);
        
        if (this->eventOnVideoStarts.length() > 0) {
        spdlog::info("[{}] motion detected, video recording begins", this->deviceName);
        string commandOnVideoStarts = regex_replace(
            this->eventOnVideoStarts, regex("\\{\\{timestamp\\}\\}"), *timestampOnVideoStarts
        );
        system((commandOnVideoStarts + " &").c_str());
        spdlog::info("[{}] onVideoStarts: command [{}] executed", this->deviceName, commandOnVideoStarts);
        } else {
        spdlog::info("[{}] onVideoStarts: no command to execute", this->deviceName);
        }
    }
}

void deviceManager::getLiveImage(vector<uint8_t>& pl) {    

    pthread_mutex_lock(&mutexLiveImage);
    pl = encodedJpgImage;
    pthread_mutex_unlock(&mutexLiveImage);

}

void deviceManager::InternalThreadEntry() {

    Mat prevFrame, currFrame, dispFrame, diffFrame;
    bool result = false;
    bool isShowingBlankFrame = false;
    VideoCapture cap;
    float rateOfChange = 0.0;
    string timestampOnVideoStarts = "";

    result = cap.open(this->deviceUri);
    spdlog::info("[{}] cap.open({}): {}", this->deviceName, this->deviceUri, result);
    uint64_t totalFrameCount = 0;
    uint32_t videoFrameCount = 0;
    FILE *ffmpegPipe = nullptr;
    int cooldown = 0;
    
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G')); 
    // Thought about moving CAP_PROP_FOURCC to config file. But seems they are good just to be hard-coded?
    if (this->framePreferredWidth > 0) { cap.set(CAP_PROP_FRAME_WIDTH, this->framePreferredWidth); }
    if (this->framePreferredHeight > 0) { cap.set(CAP_PROP_FRAME_HEIGHT, this->framePreferredHeight); }
    if (this->framePreferredFps > 0) { cap.set(CAP_PROP_FPS, this->framePreferredFps); }
    
    while (*(this->done) == 0) {
        result = cap.grab();
        if (this->skipThisFrame() == true) { continue; }
        if (result) { result = result && cap.retrieve(currFrame); }

        if (result == false || currFrame.empty() || cap.isOpened() == false) {
            spdlog::error("[{}] Unable to cap.read() a new frame. "
                "currFrame.empty(): {}, cap.isOpened(): {}. "
                "Sleep for 2 sec than then re-open()...",
                this->deviceName, currFrame.empty(), cap.isOpened());
            isShowingBlankFrame = true;
            this_thread::sleep_for(2000ms); // Don't wait for too long, most of the time the device can be re-open()ed immediately
            cap.open(this->deviceUri);
            if (this->framePreferredWidth >0 && this->framePreferredHeight > 0) {
                currFrame = Mat(this->framePreferredHeight, this->framePreferredWidth, CV_8UC3, Scalar(128, 128, 128));
            }
            else {
                // We cant just do this and skip framePreferredWidth and framePreferredHeight
                // problem will occur when piping frames to ffmpeg: In ffmpeg, we pre-define the frame size, which is mostly
                // framePreferredWidth x framePreferredHeight. If the video device is down and we supply a smaller frame, 
                // ffmpeg will wait until there are enough pixels filling the original resolution to write one frame, 
                // causing screen tearing
                currFrame = Mat(540, 960, CV_8UC3, Scalar(128, 128, 128));
            }
            // 960x540, 1280x760, 1920x1080 all have 16:9 aspect ratio.
        } else {
            if (isShowingBlankFrame == true) {
                spdlog::info("[{}] Device is back online", this->deviceName);
            }
            isShowingBlankFrame = false;
        }
        if (this->frameRotation != -1 && isShowingBlankFrame == false) { rotate(currFrame, currFrame, this->frameRotation); }

        if (totalFrameCount % this->diffFrameInterval == 0) {
            // profiling shows this if() block takes around 1-2 ms
            rateOfChange = this->getFrameChanges(prevFrame, currFrame, &diffFrame);
            prevFrame = currFrame.clone();
        }
        
        dispFrame = currFrame.clone();
        if (this->enableContoursDrawing) {
            this->overlayContours(dispFrame, diffFrame); // CPU-intensive! Use with care!
        }
        this->overlayChangeRate(dispFrame, rateOfChange, cooldown, videoFrameCount);
        this->overlayDatetime(dispFrame);    
        this->overlayDeviceName(dispFrame);
        
        totalCount = totalFrameCount;
        if (totalFrameCount % this->snapshotFrameInterval == 0) {
            pthread_mutex_lock(&mutexLiveImage);
            vector<int> configs = {};
            imencode(".jpg", dispFrame, encodedJpgImage, configs);
            pthread_mutex_unlock(&mutexLiveImage);
            // https://stackoverflow.com/questions/7054844/is-rename-atomic
            // https://stackoverflow.com/questions/29261648/atomic-writing-to-file-on-linux
            ofstream fout(this->snapshotPath + ".tmp", ios::out | ios::binary);
            fout.write((char*)encodedJpgImage.data(), encodedJpgImage.size());
            fout.close();
            rename((this->snapshotPath + ".tmp").c_str(), this->snapshotPath.c_str());

            // printf("[%s] totalFrameCount == %lu, imwrite()'ed\n", this->deviceName.c_str(), totalFrameCount);
            // imwrite() turns out to be a very expensive operation, takes up to 30 ms to finish even with ramdisk used!!!
            // profiling shows that using ramdisk isn't really helpful--perhaps imwrite()/Linux is already using RAM caching.
        }
        
        if (rateOfChange > this->rateOfChangeLower && rateOfChange < this->rateOfChangeUpper) {
        this->rateOfChangeInRange(&ffmpegPipe, &cooldown, &timestampOnVideoStarts);
        }
        
        ++totalFrameCount;
        if (cooldown >= 0) {
            cooldown --;
            if (cooldown > 0) { ++videoFrameCount; }
        }
        if (videoFrameCount >= this->maxFramesPerVideo) { cooldown = 0; }
        if (cooldown == 0) { 
            this->coolDownReachedZero(&ffmpegPipe, &videoFrameCount, &timestampOnVideoStarts);
        }  

        if (ffmpegPipe != nullptr) {
        fwrite(dispFrame.data, 1, dispFrame.dataend - dispFrame.datastart, ffmpegPipe);
        // formula of dispFrame.dataend - dispFrame.datastart height x width x channel bytes.
        // For example, for conventional 1920x1080x3 videos, one frame occupies 1920*1080*3 = 6,220,800 bytes or 6,075 KB
        // profiling shows that:
        // fwrite() takes around 10ms for piping raw video to 1080p@30fps.
        // fwirte() takes around 20ms for piping raw video to 1080p@30fps + 360p@30fps concurrently
        if (ferror(ffmpegPipe)) {
            spdlog::error("[{}] ferror(ffmpegPipe) is true, unable to fwrite() more frames to the pipe (cooldown: {})", this->deviceName, cooldown);
        }
        }
    }
    cap.release();
    spdlog::info("[{}] stop_signal received, thread startMotionDetection() quits gracefully", this->deviceName);
}
