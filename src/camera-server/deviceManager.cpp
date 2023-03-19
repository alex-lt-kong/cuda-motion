#include <arpa/inet.h>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <errno.h>
#include <fstream>
#include <limits.h>
#include <iomanip>
#include <regex>
#include <sstream>
#include <sys/socket.h>

#include <spdlog/spdlog.h>

#include "deviceManager.h"

using sysclock_t = std::chrono::system_clock;

string deviceManager::getCurrentTimestamp() {
    time_t now = sysclock_t::to_time_t(sysclock_t::now());
    //"19700101_000000"
    char buf[16] = { 0 };
    strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", localtime(&now));
    // https://www.cplusplus.com/reference/ctime/strftime/
    return string(buf);
}

deviceManager::deviceManager() {
    if (pthread_mutex_init(&mutexLiveImage, NULL) != 0) {
        throw runtime_error("pthread_mutex_init() failed, errno: " +
            to_string(errno));
    }
    spdlog::set_pattern("%Y-%m-%dT%T.%e%z|%5t|%8l| %v");
}

string deviceManager::evaluateVideoSpecficVariables(
    basic_string<char> originalString) {
    string filledString = regex_replace(originalString,
        regex(R"(\{\{timestampOnVideoStarts\}\})"), timestampOnVideoStarts);    
    filledString = regex_replace(filledString,
        regex(R"(\{\{timestampOnDeviceOffline\}\})"), timestampOnDeviceOffline);
    filledString = regex_replace(filledString,
        regex(R"(\{\{timestamp\}\})"), getCurrentTimestamp());
    return filledString;
}

string deviceManager::evaluateStaticVariables(basic_string<char> originalString) {
    string filledString = regex_replace(originalString,
        regex(R"(\{\{deviceIndex\}\})"), to_string(deviceIndex));
    filledString = regex_replace(filledString,
        regex(R"(\{\{deviceName\}\})"), conf["name"].get<string>());
    return filledString;
}

void deviceManager::setParameters(const size_t deviceIndex,
    const njson& defaultConf, njson& overrideConf) {

    // Most config items will be directly used from njson object, however,
    // given performance concern, for items that are on the critical path,
    // we will duplicate them as class member variables
    this->deviceIndex = deviceIndex;
    conf = overrideConf;
    if (!conf.contains("/name"_json_pointer))
        conf["name"] = defaultConf["name"];
    conf["name"] = evaluateStaticVariables(conf["name"]);  
    deviceName = conf["name"];
    if (!conf.contains("/uri"_json_pointer))
        conf["uri"] = defaultConf["uri"];
    conf["uri"] = evaluateStaticVariables(conf["uri"]);    

    // ===== frame =====
    if (!conf.contains("/frame/rotation"_json_pointer))
        conf["frame"]["rotation"] = defaultConf["frame"]["rotation"];
    frameRotation = conf["frame"]["rotation"];
    if (!conf.contains("/frame/preferredWidth"_json_pointer))
        conf["frame"]["preferredWidth"] = defaultConf["frame"]["preferredWidth"];
    if (!conf.contains("/frame/preferredHeight"_json_pointer))
        conf["frame"]["preferredHeight"] = defaultConf["frame"]["preferredHeight"];
    if (!conf.contains("/frame/preferredFps"_json_pointer))
        conf["frame"]["preferredFps"] = defaultConf["frame"]["preferredFps"];
    if (!conf.contains("/frame/throttleFpsIfHigherThan"_json_pointer))
        conf["frame"]["throttleFpsIfHigherThan"] =
        defaultConf["frame"]["throttleFpsIfHigherThan"];
    throttleFpsIfHigherThan = conf["frame"]["throttleFpsIfHigherThan"];
    if (!conf.contains("/frame/textOverlay/enabled"_json_pointer))
        conf["frame"]["textOverlay"]["enabled"] =
            defaultConf["frame"]["textOverlay"]["enabled"];
    textOverlayEnabled = conf["frame"]["textOverlay"]["enabled"];
    if (!conf.contains("/frame/textOverlay/fontScale"_json_pointer))
        conf["frame"]["textOverlay"]["fontScale"] =
            defaultConf["frame"]["textOverlay"]["fontScale"];
    textOverlayFontSacle = conf["frame"]["textOverlay"]["fontScale"];
    

    // =====  snapshot =====
    cout << !conf.contains("/snapshot/frameInterval"_json_pointer) << endl;
    if (!conf.contains("/snapshot/frameInterval"_json_pointer)) {
        conf["snapshot"]["frameInterval"] =
            defaultConf["snapshot"]["frameInterval"];
    }
    snapshotFrameInterval = conf["snapshot"]["frameInterval"];
    if (!conf.contains("/snapshot/ipc/switch/http"_json_pointer)) {
        conf["snapshot"]["ipc"]["switch"]["http"] =
            defaultConf["snapshot"]["ipc"]["switch"]["http"];
    }
    snapshotIpcHttpEnabled = conf["snapshot"]["ipc"]["switch"]["http"];
    if (!conf.contains("/snapshot/ipc/switch/file"_json_pointer)) {
        conf["snapshot"]["ipc"]["switch"]["file"] =
            defaultConf["snapshot"]["ipc"]["switch"]["file"];
    }
    snapshotIpcFileEnabled = conf["snapshot"]["ipc"]["switch"]["file"];
    if (snapshotIpcFileEnabled) {
        if (!conf.contains("/snapshot/ipc/file/path"_json_pointer)) {
            conf["snapshot"]["ipc"]["file"]["path"] =
                defaultConf["snapshot"]["ipc"]["file"]["path"];
        }      
        snapshotIpcFilePath = evaluateStaticVariables(
            conf["snapshot"]["ipc"]["file"]["path"]);
    }
    

    // ===== events =====
    if (!conf.contains("/events/onVideoStarts"_json_pointer))
        conf["events"]["onVideoStarts"] = defaultConf["events"]["onVideoStarts"];
    for (size_t i = 0; i < conf["events"]["onVideoStarts"].size(); ++i) {
        conf["events"]["onVideoStarts"][i] =
            evaluateStaticVariables(conf["events"]["onVideoStarts"][i]);
    }
    if (!conf.contains("/events/onVideoEnds"_json_pointer))
        conf["events"]["onVideoEnds"] = defaultConf["events"]["onVideoEnds"];    
    for (size_t i = 0; i < conf["events"]["onVideoEnds"].size(); ++i) {
        conf["events"]["onVideoEnds"][i] =
            evaluateStaticVariables(conf["events"]["onVideoEnds"][i]);
    }
    if (!conf.contains("/events/onDeviceOffline"_json_pointer))
        conf["events"]["onDeviceOffline"] =
            defaultConf["events"]["onDeviceOffline"];    
    for (size_t i = 0; i < conf["events"]["onDeviceOffline"].size(); ++i) {
        conf["events"]["onDeviceOffline"][i] =
            evaluateStaticVariables(conf["events"]["onDeviceOffline"][i]);
    }
    if (!conf.contains("/events/onDeviceBackOnline"_json_pointer))
        conf["events"]["onDeviceBackOnline"] =
            defaultConf["events"]["onDeviceBackOnline"];    
    for (size_t i = 0; i < conf["events"]["onDeviceBackOnline"].size(); ++i) {
        conf["events"]["onDeviceBackOnline"][i] =
            evaluateStaticVariables(conf["events"]["onDeviceBackOnline"][i]);
    }

    // ===== motion detection =====
    if (!conf.contains("/motionDetection/mode"_json_pointer))
        conf["motionDetection"]["mode"] = defaultConf["motionDetection"]["mode"];
    motionDetectionMode = conf["motionDetection"]["mode"];
    if (!conf.contains(
        "/motionDetection/frameDiffPercentageLowerLimit"_json_pointer))
        conf["motionDetection"]["frameDiffPercentageLowerLimit"] =
            defaultConf["motionDetection"]["frameDiffPercentageLowerLimit"];
    frameDiffPercentageLowerLimit =
        conf["motionDetection"]["frameDiffPercentageLowerLimit"];
    if (!conf.contains(
        "/motionDetection/frameDiffPercentageUpperLimit"_json_pointer))
        conf["motionDetection"]["frameDiffPercentageUpperLimit"] =
            defaultConf["motionDetection"]["frameDiffPercentageUpperLimit"];
    frameDiffPercentageUpperLimit =
        conf["motionDetection"]["frameDiffPercentageUpperLimit"];
    if (!conf.contains(
        "/motionDetection/pixelDiffAbsThreshold"_json_pointer))
        conf["motionDetection"]["pixelDiffAbsThreshold"] =
            defaultConf["motionDetection"]["pixelDiffAbsThreshold"];
    pixelDiffAbsThreshold = conf["motionDetection"]["pixelDiffAbsThreshold"];
    if (!conf.contains(
        "/motionDetection/diffEveryNthFrame"_json_pointer))
        conf["motionDetection"]["diffEveryNthFrame"] =
            defaultConf["motionDetection"]["diffEveryNthFrame"];
    diffEveryNthFrame = conf["motionDetection"]["diffEveryNthFrame"];
    if (diffEveryNthFrame == 0) {
        throw invalid_argument("diffEveryNthFrame must be greater than 0");
    }
    if (!conf.contains("/motionDetection/videoRecording/minFramesPerVideo"_json_pointer))
        conf["motionDetection"]["videoRecording"]["minFramesPerVideo"] = 
            defaultConf["motionDetection"]["videoRecording"]["minFramesPerVideo"];
    if (!conf.contains("/motionDetection/videoRecording/maxFramesPerVideo"_json_pointer))
        conf["motionDetection"]["videoRecording"]["maxFramesPerVideo"] =
            defaultConf["motionDetection"]["videoRecording"]["maxFramesPerVideo"];
    maxFramesPerVideo =
        conf["motionDetection"]["videoRecording"]["maxFramesPerVideo"];
    if (!conf.contains("/motionDetection/videoRecording/precaptureFrames"_json_pointer))
        conf["motionDetection"]["videoRecording"]["precaptureFrames"] =
            defaultConf["motionDetection"]["videoRecording"]["precaptureFrames"];
    precaptureFrames = conf["motionDetection"]["videoRecording"]["precaptureFrames"];

    if (!conf.contains("/motionDetection/videoRecording/encoder"_json_pointer))
        conf["motionDetection"]["videoRecording"]["encoder"] =
            defaultConf["motionDetection"]["videoRecording"]["encoder"];
    encoderUseExternal =
        conf["motionDetection"]["videoRecording"]["encoder"]["useExternal"];
    if (encoderUseExternal) {
        pipeRawVideoTo =
            conf["motionDetection"]["videoRecording"]["encoder"]["external"]["pipeRawVideoTo"];
    }
    if (!conf.contains("/motionDetection/drawContours"_json_pointer))
        conf["motionDetection"]["drawContours"] =
            defaultConf["motionDetection"]["drawContours"];
    drawContours = conf["motionDetection"]["drawContours"];
    
    spdlog::info("{}-th device to be used with the following configs:\n{}",
        deviceIndex, conf.dump(2));
}

deviceManager::~deviceManager() {
    pthread_mutex_destroy(&mutexLiveImage);
}

void deviceManager::overlayDatetime(Mat& frame) {
    time_t now;
    time(&now);
    //char buf[sizeof "1970-01-01 00:00:00"];
    //strftime(buf, sizeof buf, "%F %T", localtime(&now));
    string ts = getCurrentTimestamp();
    if (timestampOnDeviceOffline.size() > 0) {
        ts += " (Offline since " + timestampOnDeviceOffline + ")";
    }
    cv::Size textSize = getTextSize(ts, FONT_HERSHEY_DUPLEX,
        textOverlayFontSacle, 8 * textOverlayFontSacle, nullptr);
    putText(frame, ts, Point(5, textSize.height * 1.05), FONT_HERSHEY_DUPLEX,
        textOverlayFontSacle, Scalar(0,  0,  0  ), 8 * textOverlayFontSacle,
        LINE_AA, false);
    putText(frame, ts, Point(5, textSize.height * 1.05), FONT_HERSHEY_DUPLEX,
        textOverlayFontSacle, Scalar(255,255,255), 2 * textOverlayFontSacle,
        LINE_AA, false);
    /*
    void cv::putText 	(InputOutputArray  	img,
                        const String &  	text,
                        Point  	org,
                        int  	fontFace,
                        double  	textOverlayFontSacle,
                        Scalar  	color,
                        int  	thickness = 1,
                        int  	lineType = LINE_8,
                        bool  	bottomLeftOrigin = false 
        ) 	
    */
}

float deviceManager::getFrameChanges(Mat& prevFrame, Mat& currFrame,
    Mat* diffFrame) {
    if (prevFrame.empty() || currFrame.empty()) { return -1; }
    if (prevFrame.cols != currFrame.cols || prevFrame.rows != currFrame.rows) {
        return -1;
    }
    if (prevFrame.cols == 0 || prevFrame.rows == 0) { return -1; }
    
    absdiff(prevFrame, currFrame, *diffFrame);
    cvtColor(*diffFrame, *diffFrame, COLOR_BGR2GRAY);
    threshold(*diffFrame, *diffFrame, pixelDiffAbsThreshold, 255, THRESH_BINARY);
    int nonZeroPixels = countNonZero(*diffFrame);
    return 100.0 * nonZeroPixels / (diffFrame->rows * diffFrame->cols);
}

void deviceManager::overlayDeviceName(Mat& frame) {

    cv::Size textSize = getTextSize(deviceName, FONT_HERSHEY_DUPLEX,
        textOverlayFontSacle, 8 * textOverlayFontSacle, nullptr);
    putText(frame, deviceName,
        Point(frame.cols - textSize.width * 1.05, frame.rows - 5),
        FONT_HERSHEY_DUPLEX, textOverlayFontSacle,
        Scalar(0,  0,  0  ), 8 * textOverlayFontSacle, LINE_AA, false);
    putText(frame, deviceName,
        Point(frame.cols - textSize.width * 1.05, frame.rows - 5),
        FONT_HERSHEY_DUPLEX, textOverlayFontSacle, Scalar(255,255,255),
        2 * textOverlayFontSacle, LINE_AA, false);
}

void deviceManager::overlayStats(Mat& frame, float changeRate,
    int cooldown, long long int videoFrameCount) {
    int64_t msSinceEpoch = chrono::duration_cast<chrono::milliseconds>(
        chrono::system_clock::now().time_since_epoch()).count();
    ostringstream textToOverlay;
    if (motionDetectionMode == MODE_DETECT_MOTION) {
        textToOverlay << fixed << setprecision(2) << changeRate << "%, ";
    }
    textToOverlay << "fps: " << fixed << setprecision(1)
                  << getCurrentFps(msSinceEpoch) << " ";
    if (motionDetectionMode != MODE_DISABLED) {
        textToOverlay << "(" << cooldown << ", "
                      << maxFramesPerVideo - videoFrameCount << ")";
    }
    putText(frame, textToOverlay.str(), 
        Point(5, frame.rows-5), FONT_HERSHEY_DUPLEX, textOverlayFontSacle,
        Scalar(0,   0,   0  ), 8 * textOverlayFontSacle, LINE_AA, false);
    putText(frame, textToOverlay.str(),
        Point(5, frame.rows-5), FONT_HERSHEY_DUPLEX, textOverlayFontSacle,
            Scalar(255, 255, 255), 2 * textOverlayFontSacle, LINE_AA, false);
}

void deviceManager::overlayContours(Mat& dispFrame, Mat& diffFrame) {
    if (diffFrame.empty()) return;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(diffFrame, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE );
    int idx = 0;
    for( ; idx >= 0; idx = hierarchy[idx][0] ) {
        cv::drawContours(dispFrame, contours, idx,
            Scalar(255, 255, 255), 0.25, 8, hierarchy);
    }
}

float deviceManager::getCurrentFps(int64_t msSinceEpoch) {
    float fps = FLT_MAX;
    if (msSinceEpoch - frameTimestamps.front() > 0) {
        fps = 1000.0 * frameTimestamps.size() / (msSinceEpoch -
            frameTimestamps.front());
    }
    return fps;
}

bool deviceManager::shouldFrameBeThrottled() {

    constexpr int sampleMsUpperLimit = 60 * 1000;
    int64_t msSinceEpoch = chrono::duration_cast<chrono::milliseconds>(
        chrono::system_clock::now().time_since_epoch()).count();
    if (frameTimestamps.size() <= 1) { 
        frameTimestamps.push(msSinceEpoch); 
        return false;
    }
    
    // time complexity of frameTimestamps.size()/front()/back()/push()/pop()
    // are guaranteed to be O(1)
    if (msSinceEpoch - frameTimestamps.front() > sampleMsUpperLimit) {
        frameTimestamps.pop();
    }
    // The logic is this:
    // A timestamp is only added to the queue if it is not throttled.
    // That is, if a frame is throttled, it will not be taking into account
    // when we calculate Fps.
    if (getCurrentFps(msSinceEpoch) > throttleFpsIfHigherThan) {
        return true;
    }
    frameTimestamps.push(msSinceEpoch);  
    return false;
}

void deviceManager::asyncExecCallback(void* This, string stdout, string stderr,
    int rc) {
    if (stdout.size() > 0) {
        spdlog::info("[{}] non-empty stdout from command: [{}]",
            reinterpret_cast<deviceManager*>(This)->deviceName, stdout);    
    }
    if (stderr.size() > 0) {
        spdlog::info("[{}] non-empty stderr from command: [{}]",
            reinterpret_cast<deviceManager*>(This)->deviceName, stderr, rc);    
    }
    if (rc != 0) {
        spdlog::info("[{}] non-zero return code from command: [{}]",
            reinterpret_cast<deviceManager*>(This)->deviceName, rc);    
    }
    
}

void deviceManager::stopVideoRecording(FILE*& extRawVideoPipePtr,
    VideoWriter& vwriter, uint32_t& videoFrameCount, int cooldown) {

    auto handleOnVideoEnds = [&] () {
        if (cooldown > 0) {
            spdlog::warn("[{}] video recording stopped before cooldown "
                "reaches 0", deviceName);
        }
        
        if (conf["events"]["onVideoEnds"].size() > 0 && cooldown == 0) {
            vector<string> args;
            args.reserve(conf["events"]["onVideoEnds"].size());
            spdlog::info("[{}] video recording ends", deviceName);
            for (size_t i = 0; i < conf["events"]["onVideoEnds"].size(); ++i) {
                args.push_back(evaluateVideoSpecficVariables(
                    conf["events"]["onVideoEnds"][i]));
            }
            execAsync((void*)this, args, asyncExecCallback);
            spdlog::info("[{}] onVideoEnds triggered, command [{}] executed",
                deviceName, args[0]);
        } else if (conf["events"]["onVideoEnds"].size() > 0 && cooldown > 0) {
            spdlog::warn("[{}] onVideoEnds event defined but it won't be "
                "triggered", deviceName);
        } else {
            spdlog::info("[{}] onVideoEnds, no command to execute", deviceName);
        }
    };

    if (!encoderUseExternal) {
        vwriter.release();
    } else {
        pclose(extRawVideoPipePtr); 
        extRawVideoPipePtr = nullptr;
    }
    handleOnVideoEnds();
    videoFrameCount = 0;
}


void deviceManager::startOrKeepVideoRecording(FILE*& extRawVideoPipePtr,
    VideoWriter& vwriter, int64_t& cooldown) {

    auto handleOnVideoStarts = [&] () {
        if (conf["events"]["onVideoStarts"].size() > 0) {
            vector<string> args;
            args.reserve(conf["events"]["onVideoStarts"].size());
            for (size_t i = 0; i < conf["events"]["onVideoStarts"].size(); ++i) {
                args.push_back(
                    evaluateVideoSpecficVariables(
                        conf["events"]["onVideoStarts"][i]));
            }
            spdlog::info("[{}] motion detected, video recording begins", deviceName);
            execAsync((void*)this, args, asyncExecCallback);
            spdlog::info("[{}] onVideoStarts: command [{}] executed",
                deviceName, args[0]);
        } else {
            spdlog::info("[{}] onVideoStarts: no command to execute",
                deviceName);
        }
    };

    cooldown = conf["motionDetection"]["videoRecording"]["minFramesPerVideo"];

    if (!encoderUseExternal && vwriter.isOpened()) return;
    if (encoderUseExternal && extRawVideoPipePtr != nullptr) return;
    string command = encoderUseExternal ?
        conf["motionDetection"]["videoRecording"]["encoder"]["external"]["pipeRawVideoTo"] :
        conf["motionDetection"]["videoRecording"]["encoder"]["internal"]["videoPath"];
    timestampOnVideoStarts = getCurrentTimestamp();
    command = evaluateVideoSpecficVariables(command);
    if (!encoderUseExternal) {
        // Use OpenCV to encode video
        vwriter = VideoWriter(
            command,
            VideoWriter::fourcc('a','v','c','1'),
            conf["motionDetection"]["videoRecording"]["encoder"]["internal"]["fps"],
            Size(
                conf["motionDetection"]["videoRecording"]["encoder"]["internal"]["width"],
                conf["motionDetection"]["videoRecording"]["encoder"]["internal"]["height"]
            ));
    } else {
        // Use external encoder to encode video
        extRawVideoPipePtr = popen((command).c_str(), "w");
        if (extRawVideoPipePtr == nullptr) {
            // most likely due to invalid command or lack of memory
            throw runtime_error("popen() failed, errno: " + to_string(errno));
        }
    }
    handleOnVideoStarts();
}

void deviceManager::getLiveImage(vector<uint8_t>& pl) {
    if (encodedJpgImage.size() > 0) {
        pthread_mutex_lock(&mutexLiveImage);
        pl = encodedJpgImage;
        pthread_mutex_unlock(&mutexLiveImage);
    } else {
        pl = vector<uint8_t>();
    }
}

void deviceManager::generateBlankFrameAt1Fps(Mat& currFrame,
    const Size& actualFrameSize) {
    this_thread::sleep_for(999ms); // Throttle the generation at 1 fps.
    if (actualFrameSize.width > 0 && actualFrameSize.height > 0) {
        currFrame = Mat(actualFrameSize.height, actualFrameSize.width,
            CV_8UC3, Scalar(128, 128, 128));
    }
    else {
        // We cant just do this and skip framePreferredWidth and framePreferredHeight
        // problem will occur when piping frames to ffmpeg: In ffmpeg, we pre-define the frame size, which is mostly
        // framePreferredWidth x framePreferredHeight. If the video device is down and we supply a smaller frame, 
        // ffmpeg will wait until there are enough pixels filling the original resolution to write one frame, 
        // causing screen tearing
        currFrame = Mat(540, 960, CV_8UC3, Scalar(128, 128, 128));
        // 960x540, 1280x760, 1920x1080 all have 16:9 aspect ratio.
    }
}

void deviceManager::updateVideoCooldownAndVideoFrameCount(int64_t& cooldown,
    uint32_t& videoFrameCount) {
    if (cooldown >= 0) {
        cooldown --;
        if (cooldown > 0) { ++videoFrameCount; }
    }
    if (videoFrameCount >= maxFramesPerVideo) {
        cooldown = 0;
    }
}

void deviceManager::deviceIsOffline(Mat& currFrame, const Size& actualFrameSize,
    bool& isShowingBlankFrame) {
    
    if (isShowingBlankFrame == false) {
        timestampOnDeviceOffline = getCurrentTimestamp();
        isShowingBlankFrame = true;
        if (conf["events"]["onDeviceOffline"].size() > 0) {
            vector<string> args;
            args.reserve(conf["events"]["onDeviceOffline"].size());
            for (size_t i = 0; i < conf["events"]["onDeviceOffline"].size(); ++i) {
                args.push_back(evaluateVideoSpecficVariables(
                    conf["events"]["onDeviceOffline"][i]));
            }
            execAsync((void*)this, args, asyncExecCallback);
            spdlog::info("[{}] onDeviceOffline: command [{}] executed",
                deviceName, args[0]);
        } else {
            spdlog::info("[{}] onDeviceOffline: no command to execute",
                deviceName);
        }
    }
    generateBlankFrameAt1Fps(currFrame, actualFrameSize);
}

void deviceManager::deviceIsBackOnline(size_t& openRetryDelay,
    bool& isShowingBlankFrame) {
    spdlog::info("[{}] Device is back online", deviceName);
    timestampOnDeviceOffline = "";
    openRetryDelay = 1;
    isShowingBlankFrame = false;
    if (conf["events"]["onDeviceBackOnline"].size() > 0) {
        vector<string> args;
        args.reserve(conf["events"]["onDeviceBackOnline"].size());
        for (size_t i = 0; i < conf["events"]["onDeviceBackOnline"].size(); ++i) {
            args.push_back(evaluateVideoSpecficVariables(
                conf["events"]["onDeviceBackOnline"][i]));
        }
        execAsync((void*)this, args, asyncExecCallback);
        spdlog::info("[{}] onDeviceBackOnline: command [{}] executed",
            deviceName, args[0]);
    } 
}

void deviceManager::initializeDevice(VideoCapture& cap, bool&result,
    const Size& actualFrameSize) {
    result = cap.open(conf["uri"].get<string>());
    spdlog::info("[{}] cap.open({}): {}", deviceName, conf["uri"].get<string>(),
        result);
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G')); 
    // Thought about moving CAP_PROP_FOURCC to config file. But seems they are good just to be hard-coded?
    if (actualFrameSize.width > 0)
        cap.set(CAP_PROP_FRAME_WIDTH, actualFrameSize.width);
    if (actualFrameSize.height > 0)
        cap.set(CAP_PROP_FRAME_HEIGHT, actualFrameSize.height);
    if (conf["frame"]["preferredFps"] > 0)
        cap.set(CAP_PROP_FPS, conf["frame"]["preferredFps"]);
}

void deviceManager::prepareDataForIpc(queue<cv::Mat>& dispFrames) {

    pthread_mutex_lock(&mutexLiveImage);            
    //vector<int> configs = {IMWRITE_JPEG_QUALITY, 80};
    vector<int> configs = {};
    imencode(".jpg", dispFrames.front(), encodedJpgImage, configs);
    pthread_mutex_unlock(&mutexLiveImage);

    // Profiling show that the above mutex section without actual
    // waiting takes ~30 ms to complete, means that the CPU can only
    // handle ~30 fps

    // https://stackoverflow.com/questions/7054844/is-rename-atomic
    // https://stackoverflow.com/questions/29261648/atomic-writing-to-file-on-linux            
    if (snapshotIpcFileEnabled) {
        string sp = evaluateVideoSpecficVariables(snapshotIpcFilePath);
        ofstream fout(sp + ".tmp", ios::out | ios::binary);
        fout.write((char*)encodedJpgImage.data(), encodedJpgImage.size());
        fout.close();
        rename((sp + ".tmp").c_str(), sp.c_str());
    }
    // profiling shows from ofstream fout()... to rename() takes
    // less than 1 ms.
}

void deviceManager::InternalThreadEntry() {

    queue<Mat> dispFrames;
    Mat prevFrame, currFrame, diffFrame;
    bool result = false;
    bool isShowingBlankFrame = false;
    VideoCapture cap;
    float rateOfChange = 0.0;
    
    uint64_t retrievedFrameCount = 0;
    uint32_t videoFrameCount = 0;
    Size actualFrameSize = Size(conf["frame"]["preferredWidth"],
        conf["frame"]["preferredHeight"]);
    FILE *extRawVideoPipePtr = nullptr;
    VideoWriter vwriter;
    int64_t cooldown = 0;
    size_t openRetryDelay = 1;
    
    goto entryPoint;
    /* We use the evil goto so that we can avoid the duplication of
       initializeDevice()...*/
    
    while (!_internalThreadShouldQuit) {
        // profiling shows that cap.grab() can also be an expensive operation
        // if the source uses HTTP protocol.
        result = cap.grab();
        
        if (shouldFrameBeThrottled()) {continue; }
        if (result) {
            result = result && cap.retrieve(currFrame);
        }
        ++retrievedFrameCount;

        if (result == false || currFrame.empty() || cap.isOpened() == false) {
            deviceIsOffline(currFrame, actualFrameSize, isShowingBlankFrame);
            if (retrievedFrameCount % openRetryDelay == 0) {
                openRetryDelay *= 2;
                spdlog::error("[{}] Unable to cap.read() a new frame. "
                    "currFrame.empty(): {}, cap.isOpened(): {}. "
                    "Wait for {} frames than then re-open()...",
                    deviceName, currFrame.empty(), cap.isOpened(),
                    openRetryDelay);
entryPoint:
                initializeDevice(cap, result, actualFrameSize);
                if (retrievedFrameCount == 0) { continue; }
            }
        } else {
            if (isShowingBlankFrame) {
                deviceIsBackOnline(openRetryDelay, isShowingBlankFrame);
            }
            actualFrameSize = currFrame.size();
        }
        if (frameRotation != -1 && isShowingBlankFrame == false) {
            rotate(currFrame, currFrame, frameRotation);
        }

        if (motionDetectionMode == MODE_DETECT_MOTION &&
            retrievedFrameCount % diffEveryNthFrame == 0) {
            // profiling shows this if() block takes around 1-2 ms
            rateOfChange = getFrameChanges(prevFrame, currFrame, &diffFrame);
            prevFrame = currFrame.clone();
        }
        
        dispFrames.push(currFrame.clone()); //rvalue ref!
        if (dispFrames.size() > precaptureFrames) {
            dispFrames.pop();
        }
        if (drawContours && motionDetectionMode == MODE_DETECT_MOTION) {
            overlayContours(dispFrames.back(), diffFrame);
            // CPU-intensive! Use with care!
        }
        if (textOverlayEnabled) {
            overlayStats(
                dispFrames.back(), rateOfChange, cooldown, videoFrameCount);
            overlayDatetime(dispFrames.back());    
            overlayDeviceName(dispFrames.back());
        }
        
        if ((retrievedFrameCount - 1) % snapshotFrameInterval == 0) {
            prepareDataForIpc(dispFrames);
        }

        if (motionDetectionMode == MODE_DISABLED) {
            continue;
        }
        
        if (motionDetectionMode == MODE_ALWAYS_RECORD ||
            (rateOfChange > frameDiffPercentageLowerLimit &&
             rateOfChange < frameDiffPercentageUpperLimit &&
             motionDetectionMode == MODE_DETECT_MOTION)) {
            startOrKeepVideoRecording(extRawVideoPipePtr, vwriter, cooldown);
        }
        
        updateVideoCooldownAndVideoFrameCount(cooldown, videoFrameCount);


        if (cooldown < 0) { continue; }
        if (cooldown == 0) { 
            stopVideoRecording(extRawVideoPipePtr, vwriter,
                videoFrameCount, cooldown);
            continue;
        } 
        if (!encoderUseExternal)  {
            vwriter.write(dispFrames.front());
        } else {
            fwrite(dispFrames.front().data, 1,
                dispFrames.front().dataend - dispFrames.front().datastart,
                extRawVideoPipePtr);
            // formula of dispFrame.dataend - dispFrame.datastart height x width x channel bytes.
            // For example, for conventional 1920x1080x3 videos, one frame occupies 1920*1080*3 = 6,220,800 bytes or 6,075 KB
            // profiling shows that:
            // fwrite() takes around 10ms for piping raw video to 1080p@30fps.
            // fwirte() takes around 20ms for piping raw video to 1080p@30fps + 360p@30fps concurrently
            if (ferror(extRawVideoPipePtr)) {
                spdlog::error("[{}] ferror(extRawVideoPipePtr) is true, "
                    "unable to fwrite() more frames to the pipe (cooldown: {})",
                    deviceName, cooldown);
            }
        }
    }
    if (cooldown > 0) {
        stopVideoRecording(extRawVideoPipePtr, vwriter, videoFrameCount, cooldown);
    }
    cap.release();
    spdlog::info("[{}] thread quits gracefully", deviceName);
}
