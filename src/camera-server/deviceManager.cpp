#include <arpa/inet.h>
#include <chrono>
#include <ctime>
#include <errno.h>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits.h>
#include <iomanip>
#include <regex>
#include <sstream>
#include <sys/mman.h>
#include <sys/socket.h>

#include <spdlog/spdlog.h>

#include "deviceManager.h"

using sysclock_t = std::chrono::system_clock;

string deviceManager::getCurrentTimestamp() {
    time_t now = sysclock_t::to_time_t(sysclock_t::now());
    string ts = "19700101-000000";
    strftime(ts.data(), ts.size() + 1, "%Y%m%d-%H%M%S", localtime(&now));
    // https://www.cplusplus.com/reference/ctime/strftime/
    return ts;
    // move constructor? No, most likely it is Copy elision/RVO
}

deviceManager::deviceManager(const size_t deviceIndex,
    const njson& defaultConf, njson& overrideConf) {

    setParameters(deviceIndex, defaultConf, overrideConf);
    if (snapshotIpcHttpEnabled) {
        if (pthread_mutex_init(&mutexLiveImage, NULL) != 0) {
            spdlog::error("pthread_mutex_init() failed, {}({})",
                errno, strerror(errno));
            goto err_pthread_mutex_init;
        }
    }
    if (snapshotIpcSharedMemEnabled) {
        // Should have used multiple RAII classes to handle this but...
        sharedMemName = evaluateStaticVariables(
            conf["snapshot"]["ipc"]["sharedMem"]["sharedMemName"]);
        int shmFd = shm_open(sharedMemName.c_str(), O_RDWR | O_CREAT, PERMS);
        if (shmFd < 0) {
            spdlog::error("shm_open() failed, {}({})", errno, strerror(errno));
            goto err_shmFd;
        }
        sharedMemSize = conf["snapshot"]["ipc"]["sharedMem"]["sharedMemSize"];
        if (ftruncate(shmFd, sharedMemSize) != 0) {
            spdlog::error("ftruncate() failed, {}({})", errno, strerror(errno));
            goto err_ftruncate;
        }
        memPtr = mmap(NULL, sharedMemSize, PROT_READ | PROT_WRITE,
            MAP_SHARED, shmFd, 0);
        if (memPtr == MAP_FAILED) {
            spdlog::error("mmap() failed, {}({})", errno, strerror(errno));
            goto err_mmap;
        }
        semaphoreName = evaluateStaticVariables(
                conf["snapshot"]["ipc"]["sharedMem"]["semaphoreName"]);
        semPtr = sem_open(semaphoreName.c_str(),
            O_CREAT, PERMS, SEM_INITIAL_VALUE);

        if (semPtr == SEM_FAILED) {
            spdlog::error("sem_open() failed, {}({})", errno, strerror(errno));
            goto err_sem_open;
        }
    }
    return;
err_sem_open:
    if (munmap(memPtr, sharedMemSize) != 0) {
        spdlog::error("munmap() failed: {}({}), "
            "but there is nothing we can do", errno, strerror(errno));
    }
err_mmap:
err_ftruncate:
    if (shm_unlink(sharedMemName.c_str()) != 0)
        spdlog::error("shm_unlink() failed: {}({}), "
            "but there is nothing we can do", errno, strerror(errno));
    if (close(shmFd) != 0)
        spdlog::error("close() failed: {}({}), "
            "but there is nothing we can do", errno, strerror(errno));
err_shmFd:
err_pthread_mutex_init:
    if (snapshotIpcHttpEnabled)
        if (pthread_mutex_destroy(&mutexLiveImage) != 0) {
            spdlog::error("pthread_mutex_destroy() failed:  {}({}), "
            "but there is nothing we can do", errno, strerror(errno));
        }
    throw runtime_error("initializing deviceManager instance failed. "
        "Check log for details");
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

    if (!conf.contains("/snapshot/ipc/switch/sharedMem"_json_pointer)) {
        conf["snapshot"]["ipc"]["switch"]["sharedMem"] =
            defaultConf["snapshot"]["ipc"]["switch"]["sharedMem"];
    }
    snapshotIpcSharedMemEnabled = conf["snapshot"]["ipc"]["switch"]["sharedMem"];
    if (snapshotIpcSharedMemEnabled) {
        if (!conf.contains("/snapshot/ipc/sharedMem/semaphoreName"_json_pointer)) {
            conf["snapshot"]["ipc"]["sharedMem"]["semaphoreName"] =
                defaultConf["snapshot"]["ipc"]["sharedMem"]["semaphoreName"];
        }
        if (!conf.contains("/snapshot/ipc/sharedMem/sharedMemName"_json_pointer)) {
            conf["snapshot"]["ipc"]["sharedMem"]["sharedMemName"] =
                defaultConf["snapshot"]["ipc"]["sharedMem"]["sharedMemName"];
        }
        if (!conf.contains("/snapshot/ipc/sharedMem/sharedMemSize"_json_pointer)) {
            conf["snapshot"]["ipc"]["sharedMem"]["sharedMemSize"] =
                defaultConf["snapshot"]["ipc"]["sharedMem"]["sharedMemSize"];
        }
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
    minFramesPerVideo =
        conf["motionDetection"]["videoRecording"]["minFramesPerVideo"];
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
    if (!conf.contains("/motionDetection/videoRecording/encoder/useExternal"_json_pointer))
        conf["motionDetection"]["videoRecording"]["encoder"]["useExternal"] =
            defaultConf["motionDetection"]["videoRecording"]["encoder"]["useExternal"];
    encoderUseExternal =
        conf["motionDetection"]["videoRecording"]["encoder"]["useExternal"];
    if (encoderUseExternal) {
        if (!conf.contains("/motionDetection/videoRecording/encoder/external"_json_pointer))
            conf["motionDetection"]["videoRecording"]["encoder"][
                "external"]["pipeRawVideoTo"] =
                defaultConf["motionDetection"]["videoRecording"]["encoder"][
                "external"]["pipeRawVideoTo"];
        pipeRawVideoTo = conf["motionDetection"]["videoRecording"]["encoder"][
                "external"]["pipeRawVideoTo"];
    } else {
        if (!conf.contains("/motionDetection/videoRecording/encoder/internal/videoPath"_json_pointer))
            conf["motionDetection"]["videoRecording"]["encoder"]["internal"]["videoPath"] =
                defaultConf["motionDetection"]["videoRecording"]["encoder"]["internal"]["videoPath"];
        conf["motionDetection"]["videoRecording"]["encoder"]["internal"]["videoPath"] =
            evaluateStaticVariables(conf["motionDetection"]["videoRecording"]["encoder"]["internal"]["videoPath"]);
        if (!conf.contains("/motionDetection/videoRecording/encoder/internal/fps"_json_pointer))
            conf["motionDetection"]["videoRecording"]["encoder"]["internal"]["fps"] =
                defaultConf["motionDetection"]["videoRecording"]["encoder"]["internal"]["fps"];
        if (!conf.contains("/motionDetection/videoRecording/encoder/internal/width"_json_pointer))
            conf["motionDetection"]["videoRecording"]["encoder"]["internal"]["width"] =
                defaultConf["motionDetection"]["videoRecording"]["encoder"]["internal"]["width"];
        if (!conf.contains("/motionDetection/videoRecording/encoder/internal/height"_json_pointer))
            conf["motionDetection"]["videoRecording"]["encoder"]["internal"]["height"] =
                defaultConf["motionDetection"]["videoRecording"]["encoder"]["internal"]["height"];
    }
    if (!conf.contains("/motionDetection/drawContours"_json_pointer))
        conf["motionDetection"]["drawContours"] =
            defaultConf["motionDetection"]["drawContours"];
    drawContours = conf["motionDetection"]["drawContours"];
    
    spdlog::info("{}-th device to be used with the following configs:\n{}",
        deviceIndex, conf.dump(2));
}

deviceManager::~deviceManager() {
    if (snapshotIpcHttpEnabled) {
        if (pthread_mutex_destroy(&mutexLiveImage) != 0) {
            spdlog::error("pthread_mutex_destroy() failed: {}, "
                "but there is nothing we can do", errno);
        }
    }
    if (snapshotIpcSharedMemEnabled) {
        // unlink and close() are both needed: unlink only disassociates the
        // name from the underlying semaphore object, but the semaphore object
        // is not gone. It will only be gone when we close() it.
        if (sem_unlink(semaphoreName.c_str()) != 0) {
            spdlog::error("sem_unlink() failed: {}({}), "
                "but there is nothing we can do", errno, strerror(errno));
        }
        if (sem_close(semPtr) != 0) {
            spdlog::error("sem_close() failed: {}({}), "
                "but there is nothing we can do", errno, strerror(errno));
        }
        if (munmap(memPtr, sharedMemSize) != 0) {
            spdlog::error("munmap() failed: {}({}), "
                "but there is nothing we can do", errno, strerror(errno));
        }
        if (shm_unlink(sharedMemName.c_str()) != 0) {
            spdlog::error("shm_unlink() failed: {}({}), "
                "but there is nothing we can do", errno, strerror(errno));
        }
        if (close(shmFd) != 0) {
            spdlog::error("close() failed: {}({}), "
                "but there is nothing we can do", errno, strerror(errno));
        }
    }
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
    textToOverlay << fixed << setprecision(1)
                  << getCurrentFps(msSinceEpoch) << "fps ";
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
            Scalar(255, 255, 255), 1, LINE_8, hierarchy);
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
        if (pclose(extRawVideoPipePtr) != 0) {
            spdlog::error("pclose(encoderUseExternal) error: {}({}), "
                "but there is nothing else we can do", errno, strerror(errno));
        }
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

    cooldown = minFramesPerVideo;

    // These two if's: video recording is in progress already.
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
        extRawVideoPipePtr = popen(command.c_str(), "w");
        if (extRawVideoPipePtr == NULL) {
            // most likely due to invalid command or lack of memory
            spdlog::error("[{}] popen() failed, recording won't start: {}({})",
                deviceName, errno, strerror(errno));
        }
    }
    handleOnVideoStarts();
}

void deviceManager::getLiveImage(vector<uint8_t>& pl) {
    if (encodedJpgImage.size() > 0) {
        if (pthread_mutex_lock(&mutexLiveImage) != 0) {
            spdlog::error("[{}] pthread_mutex_lock() failed, "
            "will skip live image data copy, errno: {}", deviceName, errno);
            return;
        }
        pl = encodedJpgImage;
        if (pthread_mutex_unlock(&mutexLiveImage) != 0) {
            spdlog::error("[{}] pthread_mutex_unlock() failed, errno: {}",
                deviceName, errno);
            return;
        }
    } else {
        pl = vector<uint8_t>();
    }
}

void deviceManager::generateBlankFrameAt1Fps(Mat& currFrame,
    const Size& actualFrameSize) {
    this_thread::sleep_for(999ms); // Throttle the generation at 1 fps.

    /* Even if we generate nothing but a blank screen, we cant just use some
    hardcoded values and skip framePreferredWidth/actualFrameSize.width and
    framePreferredHeight/actualFrameSize.height.
    The problem will occur when piping frames to ffmpeg: In ffmpeg, we
    pre-define the frame size, which is mostly framePreferredWidth x
    framePreferredHeight. If the video device is down and we supply a
    smaller frame, ffmpeg will wait until there are enough pixels filling
    the original resolution to write one frame, causing screen tearing
    */
    if (actualFrameSize.width > 0 && actualFrameSize.height > 0) {
        currFrame = Mat(actualFrameSize.height, actualFrameSize.width,
            CV_8UC3, Scalar(128, 128, 128));
    }
    else {
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
    // Seems CAP_PROP_FOURCC is for VideoWriter only
    //cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G')); 
    
    if (actualFrameSize.width > 0)
        cap.set(CAP_PROP_FRAME_WIDTH, actualFrameSize.width);
    if (actualFrameSize.height > 0)
        cap.set(CAP_PROP_FRAME_HEIGHT, actualFrameSize.height);
    if (conf["frame"]["preferredFps"] > 0)
        cap.set(CAP_PROP_FPS, conf["frame"]["preferredFps"]);
}

void deviceManager::prepareDataForIpc(queue<cv::Mat>& dispFrames) {
    //vector<int> configs = {IMWRITE_JPEG_QUALITY, 80};
    vector<int> configs = {};
    if (snapshotIpcHttpEnabled) {
        pthread_mutex_lock(&mutexLiveImage);
        imencode(".jpg", dispFrames.front(), encodedJpgImage, configs);
        pthread_mutex_unlock(&mutexLiveImage);
    } else {
        imencode(".jpg", dispFrames.front(), encodedJpgImage, configs);
    }

    // Profiling show that the above mutex section without actual
    // waiting takes ~30 ms to complete, means that the CPU can only
    // handle ~30 fps

    // https://stackoverflow.com/questions/7054844/is-rename-atomic
    // https://stackoverflow.com/questions/29261648/atomic-writing-to-file-on-linux            
    if (snapshotIpcFileEnabled) {
        string sp = evaluateVideoSpecficVariables(snapshotIpcFilePath);
        ofstream fout(sp + ".tmp", ios::out | ios::binary);
        if (!fout.good()) {
            spdlog::error("[{}] Failed to open file [{}]: {}",
                deviceName, sp + ".tmp", strerror(errno));
        } else {            
            fout.write((char*)encodedJpgImage.data(), encodedJpgImage.size());
            if (!fout.good()) {
                spdlog::error("[{}] Failed to write to file [{}]: {}",
                    deviceName, sp + ".tmp", strerror(errno));
            }
            fout.close();
            if (rename((sp + ".tmp").c_str(), sp.c_str()) != 0) {
                spdlog::error("[{}] Failed to rename [{}] tp [{}]: {}",
                    deviceName, sp + ".tmp", sp, strerror(errno));
            }
        }
        // profiling shows from ofstream fout()... to rename() takes
        // less than 1 ms.
    }

    if (snapshotIpcSharedMemEnabled) {
        size_t s = encodedJpgImage.size();
        if (s > sharedMemSize - sizeof(size_t)) {
            spdlog::error("[{}] encodedJpgImage({} bytes) too large for "
            "sharedMemSize({} bytes)", deviceName, s, sharedMemSize);
            s = 0;
        } else {
            if (sem_wait(semPtr) != 0) {
                spdlog::error("[{}] sem_wait() failed: {}, the program will "
                    "continue to run but the semaphore mechanism could be broken",
                    deviceName, errno);
            } else {
                
                memcpy(memPtr, &s, sizeof(size_t));
                memcpy((uint8_t*)memPtr + sizeof(size_t), encodedJpgImage.data(),
                    encodedJpgImage.size());
                if (sem_post(semPtr) != 0) {
                    spdlog::error("[{}] sem_post() failed: {}, the program will "
                        "continue to run but the semaphore mechanism could be broken",
                        deviceName, errno);
                }
            }
        }
    }
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
    
    while (ev_flag == 0) {
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
            /* fwrite() is already a buffered method, adding an extra layer
            of manual buffer isn't likely to improve performance.
            */
            size_t frameSize = dispFrames.front().dataend -
                dispFrames.front().datastart;
            if (fwrite(dispFrames.front().data, 1, frameSize,
                    extRawVideoPipePtr) != frameSize) {
                spdlog::error("[{}] fwrite() to external pipe failed: {}({})",
                    deviceName, errno, strerror(errno));
                stopVideoRecording(extRawVideoPipePtr, vwriter,
                    videoFrameCount, cooldown);

            }
            // formula of dispFrame.dataend - dispFrame.datastart height x width x channel bytes.
            // For example, for conventional 1920x1080x3 videos, one frame occupies 1920*1080*3 = 6,220,800 bytes or 6,075 KB
            // profiling shows that:
            // fwrite() takes around ~10ms for piping raw video to 1080p@30fps.
            // fwirte() takes around ~20ms for piping raw video to 1080p@30fps + 360p@30fps concurrently

        }
    }
    if (cooldown > 0) {
        stopVideoRecording(extRawVideoPipePtr, vwriter, videoFrameCount, cooldown);
    }
    cap.release();
    spdlog::info("[{}] thread quits gracefully", deviceName);
}
