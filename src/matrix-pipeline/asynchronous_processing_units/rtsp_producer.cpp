#include "rtsp_producer.h"

#include <fmt/format.h>

namespace MatrixPipeline::ProcessingUnit{

RtspProducer::RtspProducer() {
    // Constructor
}

RtspProducer::~RtspProducer() {
    // CRITICAL: We must stop the worker thread BEFORE destroying m_writer.
    // If we rely on the base class destructor (~IAsynchronousProcessingUnit), 
    // it runs AFTER ~RtspProducer, meaning the thread might try to access 
    // the destroyed m_writer, causing a segfault.
    stop(); 
    
    if (m_writer.isOpened()) {
        m_writer.release();
        SPDLOG_INFO("RTSP Writer released.");
    }
}

bool RtspProducer::init(const njson &config) {
    try {
        // 1. Parse Configuration
        std::string rtsp_url = config.value("rtsp_url", "rtsp://127.0.0.1:8554/mystream");
        int width = config.value("width", 1920);
        int height = config.value("height", 1080);
        int fps = config.value("fps", 30);
        int bitrate = config.value("bitrate_kbps", 4000);

        m_frame_size = cv::Size(width, height);
        m_fps = static_cast<double>(fps);

        // 2. Build Pipeline
        m_pipeline_string = build_pipeline(width, height, fps, bitrate, rtsp_url);
        SPDLOG_INFO("Initializing RTSP Producer with pipeline: {}", m_pipeline_string);

        // 3. Open VideoWriter (Try to connect to MediaMTX)
        // CAP_GSTREAMER is essential.
        if (!m_writer.open(m_pipeline_string, cv::CAP_GSTREAMER, 0, m_fps, m_frame_size, true)) {
            SPDLOG_ERROR("Failed to open RTSP pipeline. Is MediaMTX running at {}?", rtsp_url);
            return false;
        }

        SPDLOG_INFO("RTSP Producer initialized successfully.");
        return true;

    } catch (const std::exception &e) {
        SPDLOG_ERROR("Exception in RtspProducer::init: {}", e.what());
        return false;
    }
}

void RtspProducer::on_frame_ready(cv::cuda::GpuMat &frame, PipelineContext &ctx) {
    // Safety check
    if (!m_writer.isOpened()) {
        SPDLOG_WARN("RTSP Writer is closed, dropping frame {}", ctx.frame_seq_num);
        // Optional: Implement simple re-connect logic here if desired
        return;
    }

    try {
        // 1. Download from GPU to CPU
        // This is the unavoidable bridge between OpenCV CUDA and GStreamer appsrc
        cv::Mat cpu_frame;
        frame.download(cpu_frame);

        // 2. Write to Pipeline
        // If the pipeline is broken (server died), this might throw or print GStreamer errors
        m_writer.write(cpu_frame);

    } catch (const cv::Exception &e) {
        SPDLOG_ERROR("OpenCV error in RTSP write: {}", e.what());
    }
}

std::string RtspProducer::build_pipeline(int width, int height, int fps, int bitrate_kbps, const std::string& url) {
    // We explicitly use 'protocols=tcp' for reliability.
    // We use 'nvh264enc' for NVIDIA GPU encoding.
    // 'config-interval=1' sends SPS/PPS every second (critical for new clients joining stream).
    
    // Note: bitrate in nvh264enc is usually in bits/sec or kbits/sec depending on version.
    // In standard GStreamer nvh264enc, 'bitrate' property is in kbits/sec.
    
    return fmt::format(
        "appsrc ! "
        "videoconvert ! "
        "video/x-raw,format=I420 ! "
        "nvh264enc bitrate={} preset=low-latency-hq ! "
        "h264parse ! "
        "rtph264pay config-interval=1 pt=96 ! "
        "rtspclientsink location={} protocols=tcp",
        bitrate_kbps, url
    );
}
}