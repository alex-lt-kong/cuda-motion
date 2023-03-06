#include <arpa/inet.h>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <sys/socket.h>

#include <spdlog/spdlog.h>

#include "utils.h"
#include "deviceManager.h"

using sysclock_t = std::chrono::system_clock;

string deviceManager::getCurrentTimestamp() {
    std::time_t now = sysclock_t::to_time_t(sysclock_t::now());
    //"19700101_000000"
    char buf[16] = { 0 };
    std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", localtime(&now));
    // https://www.cplusplus.com/reference/ctime/strftime/
    return std::string(buf);
}


string deviceManager::convertToString(char* a, int size) {
    int i;
    string s = "";
    for (i = 0; i < size; i++) {
        s = s + a[i];
    }
    return s;
}

deviceManager::deviceManager() {
    if (pthread_mutex_init(&mutexLiveImage, NULL) != 0) {
        throw runtime_error("pthread_mutex_init() failed, errno: " +
            to_string(errno));
    }
}

string deviceManager::fillinVariables(basic_string<char> originalString) {
    string filledString = regex_replace(originalString,
        regex(R"(\{\{deviceIndex\}\})"), to_string(deviceIndex));
    return filledString;
}

void deviceManager::setParameters(const size_t deviceIndex,
    const njson& defaultConf, njson& overrideConf) {

    this->deviceIndex = deviceIndex;    
    conf = overrideConf;
    if (!conf.contains("uri")) conf["uri"] = defaultConf["uri"];
    conf["uri"] = fillinVariables(conf["uri"]);    

    if (!conf.contains("name")) conf["name"] = defaultConf["name"];
    conf["name"] = fillinVariables(conf["name"]);  
    deviceName = conf["name"];
    // deviceName is duplicated as it is on the critical path.

    if (!conf.contains("/frame/rotation"_json_pointer))
        conf["frame"]["rotation"] = defaultConf["frame"]["rotation"];
    framePreferredWidth = overrideConf["frame"]["preferredWidth"];
    framePreferredHeight = overrideConf["frame"]["preferredHeight"];
    framePreferredFps = overrideConf["frame"]["preferredFps"];
    if (!conf.contains("/frame/throttleFpsIfHigherThan"_json_pointer))
        conf["frame"]["throttleFpsIfHigherThan"] =
        defaultConf["frame"]["throttleFpsIfHigherThan"];
    throttleFpsIfHigherThan = conf["frame"]["throttleFpsIfHigherThan"];
    // throttleFpsIfHigherThan is duplicated as it is on the critical path.
    fontScale = overrideConf["frame"]["overlayTextFontScale"];
    if (!conf.contains("/frame/drawContours"_json_pointer))
        conf["frame"]["drawContours"] = defaultConf["frame"]["drawContours"];

    // =====  snapshot =====
    snapshotPath = overrideConf.contains("/snapshot/path"_json_pointer) ?
        overrideConf["snapshot"]["path"] : defaultConf["snapshot"]["path"];
    snapshotPath = fillinVariables(snapshotPath);
    snapshotFrameInterval = overrideConf.contains("/snapshot/frameInterval"_json_pointer) ?
        overrideConf["snapshot"]["frameInterval"] : defaultConf["snapshot"]["frameInterval"];

    // ===== events =====
    if (!conf.contains("/events/onVideoStarts"_json_pointer))
        conf["events"]["onVideoStarts"] = defaultConf["events"]["onVideoStarts"];
    conf["events"]["onVideoStarts"] = fillinVariables(conf["events"]["onVideoStarts"]);  
    if (!conf.contains("/events/onVideoEnds"_json_pointer))
        conf["events"]["onVideoEnds"] = defaultConf["events"]["onVideoEnds"];
    conf["events"]["onVideoEnds"] = fillinVariables(conf["events"]["onVideoEnds"]);  

    ffmpegCommand = overrideConf["ffmpegCommand"];
    rateOfChangeLower = overrideConf["motionDetection"]["frameLevelRateOfChangeLowerLimit"];
    rateOfChangeUpper = overrideConf["motionDetection"]["frameLevelRateOfChangeUpperLimit"];
    pixelLevelThreshold = overrideConf["motionDetection"]["pixelLevelDiffThreshold"];
    diffFrameInterval = overrideConf["motionDetection"]["diffFrameInterval"];

    // ===== video recording =====
    if (!conf.contains("/videoRecording/minFramesPerVideo"_json_pointer))
        conf["videoRecording"]["minFramesPerVideo"] = 
            defaultConf["videoRecording"]["minFramesPerVideo"];
    if (!conf.contains("/videoRecording/maxFramesPerVideo"_json_pointer))
        conf["videoRecording"]["maxFramesPerVideo"] =
            defaultConf["videoRecording"]["maxFramesPerVideo"];
    
    frameIntervalInMs = 1000 * (1.0 / throttleFpsIfHigherThan);
    
    spdlog::info("{}-th device to be used with the following configs:\n{}",
        deviceIndex, conf.dump(2));
}

deviceManager::~deviceManager() {
    pthread_mutex_destroy(&mutexLiveImage);
}

void deviceManager::overlayDatetime(Mat frame) {
    time_t now;
    time(&now);
    char buf[sizeof "1970-01-01 00:00:00"];
    strftime(buf, sizeof buf, "%F %T", localtime(&now));
    cv::Size textSize = getTextSize(buf, FONT_HERSHEY_DUPLEX, fontScale, 8 * fontScale, nullptr);
    putText(frame, buf, Point(5, textSize.height * 1.05), FONT_HERSHEY_DUPLEX, fontScale, Scalar(0,  0,  0  ), 8 * fontScale, LINE_AA, false);
    putText(frame, buf, Point(5, textSize.height * 1.05), FONT_HERSHEY_DUPLEX, fontScale, Scalar(255,255,255), 2 * fontScale, LINE_AA, false);
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
    threshold(*diffFrame, *diffFrame, pixelLevelThreshold, 255, THRESH_BINARY);
    int nonZeroPixels = countNonZero(*diffFrame);
    return 100.0 * nonZeroPixels / (diffFrame->rows * diffFrame->cols);
}

void deviceManager::overlayDeviceName(Mat frame) {

    cv::Size textSize = getTextSize(deviceName, FONT_HERSHEY_DUPLEX, fontScale, 8 * fontScale, nullptr);
    putText(frame, deviceName, Point(frame.cols - textSize.width * 1.05, frame.rows - 5), 
            FONT_HERSHEY_DUPLEX, fontScale, Scalar(0,  0,  0  ), 8 * fontScale, LINE_AA, false);
    putText(frame, deviceName, Point(frame.cols - textSize.width * 1.05, frame.rows - 5),
            FONT_HERSHEY_DUPLEX, fontScale, Scalar(255,255,255), 2 * fontScale, LINE_AA, false);
}

void deviceManager::overlayChangeRate(Mat frame, float changeRate, int cooldown, long long int videoFrameCount) {
    int value = changeRate * 100;
    stringstream ssChangeRate;
    ssChangeRate << fixed << setprecision(2) << changeRate;
    putText(frame, ssChangeRate.str() + "% (" +
        to_string(cooldown) + ", " +
        to_string(conf["videoRecording"]["maxFramesPerVideo"].get<int64_t>() - videoFrameCount) + ")", 
        Point(5, frame.rows-5), FONT_HERSHEY_DUPLEX, fontScale,
        Scalar(0,   0,   0  ), 8 * fontScale, LINE_AA, false);
    putText(frame, ssChangeRate.str() + "% (" + to_string(cooldown) + ", " +
        to_string(conf["videoRecording"]["maxFramesPerVideo"].get<int64_t>() - videoFrameCount) + ")",
        Point(5, frame.rows-5), FONT_HERSHEY_DUPLEX, fontScale, Scalar(255, 255, 255), 2 * fontScale, LINE_AA, false);
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

bool deviceManager::shouldFrameBeThrottled() {
    int sampleMsUpperLimit = 60 * 1000;
    int currMsSinceEpoch = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
    if (frameTimestamps.size() <= 1) { 
        frameTimestamps.push(currMsSinceEpoch); 
        return false;
    }
    
    float fps = 1000.0 * frameTimestamps.size() / (1 + currMsSinceEpoch - frameTimestamps.front());
    if (currMsSinceEpoch - frameTimestamps.front() > sampleMsUpperLimit) {
        frameTimestamps.pop();
    }
    if (fps > throttleFpsIfHigherThan) { return true; }
    frameTimestamps.push(currMsSinceEpoch);  
    return false;
}

void deviceManager::coolDownReachedZero(
  FILE** ffmpegPipe, uint32_t* videoFrameCount, string* timestampOnVideoStarts
) {
    if (*ffmpegPipe != nullptr) { // No, you cannot pclose() a nullptr
        pclose(*ffmpegPipe); 
        *ffmpegPipe = nullptr; 
        
        if (conf["events"]["onVideoEnds"].get<string>().length() > 0) {
            spdlog::info("[{}] video recording ends", deviceName);
            string commandOnVideoEnds = regex_replace(
                conf["events"]["onVideoEnds"].get<string>(),
                regex("\\{\\{timestamp\\}\\}"), *timestampOnVideoStarts);
            exec_async((void*)this, commandOnVideoEnds,
                [](void* This, string output){
                spdlog::info("[{}] stdout/stderr from command: [{}]",
                reinterpret_cast<deviceManager*>(This)->deviceName, output);
            });

            spdlog::info("[{}] onVideoEnds triggered, command [{}] executed",
                deviceName, commandOnVideoEnds);
        } else {
            spdlog::info("[{}] onVideoEnds, no command to execute", deviceName);
        }
    }
    *videoFrameCount = 0;
}


void deviceManager::rateOfChangeInRange(
  FILE** ffmpegPipe, int* cooldown, string* timestampOnVideoStarts
) {
    *cooldown = conf["videoRecording"]["minFramesPerVideo"];
    if (*ffmpegPipe == nullptr) {
        string command = ffmpegCommand;
        *timestampOnVideoStarts = getCurrentTimestamp();
        command = regex_replace(
        command, regex("\\{\\{timestamp\\}\\}"), *timestampOnVideoStarts
        );
        *ffmpegPipe = popen((command).c_str(), "w");
        //setvbuf(*ffmpegPipe, &(buff), _IOFBF, buff_size);
        
        if (conf["events"]["onVideoStarts"].get<string>().length() > 0) {
            spdlog::info("[{}] motion detected, video recording begins", deviceName);
            string commandOnVideoStarts = regex_replace(
                conf["events"]["onVideoStarts"].get<string>(),
                regex("\\{\\{timestamp\\}\\}"), *timestampOnVideoStarts
            );
            exec_async((void*)this, commandOnVideoStarts, [](void* This, string output){
                    spdlog::info("[{}] stdout/stderr from command: [{}]",
                    reinterpret_cast<deviceManager*>(This)->deviceName, output);
            });
            spdlog::info("[{}] onVideoStarts: command [{}] executed",
                deviceName, commandOnVideoStarts);
        } else {
            spdlog::info("[{}] onVideoStarts: no command to execute",
                deviceName);
        }
    }
}

void deviceManager::getLiveImage(vector<uint8_t>& pl) {
    pthread_mutex_lock(&mutexLiveImage);
    pl = encodedJpgImage;
    pthread_mutex_unlock(&mutexLiveImage);
}

void deviceManager::InternalThreadEntry() {

    vector<int> configs = {IMWRITE_JPEG_QUALITY, 80};
    queue<Mat> dispFrames;
    Mat prevFrame, currFrame, diffFrame;
    bool result = false;
    bool isShowingBlankFrame = false;
    VideoCapture cap;
    float rateOfChange = 0.0;
    string timestampOnVideoStarts = "";

    result = cap.open(conf["uri"].get<string>());
    spdlog::info("[{}] cap.open({}): {}", deviceName,
        conf["uri"].get<string>(), result);
    uint64_t totalFrameCount = 0;
    uint32_t videoFrameCount = 0;
    FILE *ffmpegPipe = nullptr;
    int cooldown = 0;
    
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G')); 
    // Thought about moving CAP_PROP_FOURCC to config file. But seems they are good just to be hard-coded?
    if (framePreferredWidth > 0) { cap.set(CAP_PROP_FRAME_WIDTH, framePreferredWidth); }
    if (framePreferredHeight > 0) { cap.set(CAP_PROP_FRAME_HEIGHT, framePreferredHeight); }
    if (framePreferredFps > 0) { cap.set(CAP_PROP_FPS, framePreferredFps); }
    
    while (!_internalThreadShouldQuit) {
        result = cap.grab();
        if (shouldFrameBeThrottled() == true) { continue; }
        if (result) { result = result && cap.retrieve(currFrame); }

        if (result == false || currFrame.empty() || cap.isOpened() == false) {
            spdlog::error("[{}] Unable to cap.read() a new frame. "
                "currFrame.empty(): {}, cap.isOpened(): {}. "
                "Sleep for 2 sec than then re-open()...",
                deviceName, currFrame.empty(), cap.isOpened());
            isShowingBlankFrame = true;
            this_thread::sleep_for(2000ms); // Don't wait for too long, most of the time the device can be re-open()ed immediately
            cap.open(conf["uri"].get<string>());
            if (framePreferredWidth >0 && framePreferredHeight > 0) {
                currFrame = Mat(framePreferredHeight, framePreferredWidth, CV_8UC3, Scalar(128, 128, 128));
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
                spdlog::info("[{}] Device is back online", deviceName);
            }
            isShowingBlankFrame = false;
        }
        if (conf["frame"]["rotation"] != -1 && isShowingBlankFrame == false) {
            rotate(currFrame, currFrame, conf["frame"]["rotation"]);
        }

        if (totalFrameCount % diffFrameInterval == 0) {
            // profiling shows this if() block takes around 1-2 ms
            rateOfChange = getFrameChanges(prevFrame, currFrame, &diffFrame);
            prevFrame = currFrame.clone();
        }
        
        dispFrames.push(currFrame.clone()); //rvalue ref!
        if (dispFrames.size() > 5) {
            dispFrames.pop();
        }
        if (conf["frame"]["drawContours"]) {
            overlayContours(dispFrames.back(), diffFrame);
            // CPU-intensive! Use with care!
        }
        overlayChangeRate(
            dispFrames.back(), rateOfChange, cooldown, videoFrameCount);
        overlayDatetime(dispFrames.back());    
        overlayDeviceName(dispFrames.back());
        
        if (totalFrameCount % snapshotFrameInterval == 0) {
            pthread_mutex_lock(&mutexLiveImage);            
            imencode(".jpg", dispFrames.front(), encodedJpgImage, configs);
            pthread_mutex_unlock(&mutexLiveImage);
            // Profiling show that the above mutex section without actual
            // waiting takes ~30 ms to complete, means that the CPU can only
            // handle ~30 fps

            // https://stackoverflow.com/questions/7054844/is-rename-atomic
            // https://stackoverflow.com/questions/29261648/atomic-writing-to-file-on-linux            
            ofstream fout(snapshotPath + ".tmp", ios::out | ios::binary);
            fout.write((char*)encodedJpgImage.data(), encodedJpgImage.size());
            fout.close();
            rename((snapshotPath + ".tmp").c_str(), snapshotPath.c_str());
            // profiling shows from ofstream fout()... to rename() takes
            // less than 1 ms.

        }
        
        if (rateOfChange > rateOfChangeLower && rateOfChange < rateOfChangeUpper) {
            rateOfChangeInRange(&ffmpegPipe, &cooldown, &timestampOnVideoStarts);
        }
        
        ++totalFrameCount;
        if (cooldown >= 0) {
            cooldown --;
            if (cooldown > 0) { ++videoFrameCount; }
        }
        if (videoFrameCount >= conf["videoRecording"]["maxFramesPerVideo"]) { cooldown = 0; }
        if (cooldown == 0) { 
            coolDownReachedZero(&ffmpegPipe, &videoFrameCount, &timestampOnVideoStarts);
        }  

        if (ffmpegPipe != nullptr) {
            fwrite(dispFrames.front().data, 1,
                dispFrames.front().dataend - dispFrames.front().datastart,
                ffmpegPipe);
            // formula of dispFrame.dataend - dispFrame.datastart height x width x channel bytes.
            // For example, for conventional 1920x1080x3 videos, one frame occupies 1920*1080*3 = 6,220,800 bytes or 6,075 KB
            // profiling shows that:
            // fwrite() takes around 10ms for piping raw video to 1080p@30fps.
            // fwirte() takes around 20ms for piping raw video to 1080p@30fps + 360p@30fps concurrently
            if (ferror(ffmpegPipe)) {
                spdlog::error("[{}] ferror(ffmpegPipe) is true, unable to fwrite() "
                    "more frames to the pipe (cooldown: {})",
                    deviceName, cooldown);
            }
        }
    }
    cap.release();
    spdlog::info("[{}] thread quits gracefully", deviceName);
}
