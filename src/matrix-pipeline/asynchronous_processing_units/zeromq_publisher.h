/*#pragma once

#include "frame_msg.pb.h"
#include "../interfaces/i_asynchronous_processing_unit.h"
#include "../utils.h"

#include <spdlog/spdlog.h>
#include <zmq.hpp>

#include <string>
#include <vector>
#include <memory>
#include <chrono>

namespace MatrixPipeline::ProcessingUnit {

class ZeroMqPublisher : public IAsynchronousProcessingUnit {
private:
    zmq::context_t m_ctx;
    zmq::socket_t m_socket;
    std::string m_endpoint{"tcp://*:5555"};

    // If true, we encode to JPEG before sending.
    // If false, we send raw pixel data.
    bool m_use_compression{false};

    std::unique_ptr<Utils::NvJpegEncoder> m_gpu_encoder;

    // CPU buffer for downloading from GPU in Raw mode
    cv::Mat m_cpu_buffer;

    // --- Egress Monitoring State ---
    size_t m_egress_bytes_accum{0};
    size_t m_egress_frames_accum{0};
    std::chrono::steady_clock::time_point m_last_report_time;
    // Report interval in seconds (e.g., 60 for 1 minute)
    const int m_report_interval_sec{600};

public:
    ZeroMqPublisher() : m_socket(m_ctx, zmq::socket_type::pub) {
        m_last_report_time = std::chrono::steady_clock::now();
    }

    ~ZeroMqPublisher() override {
        stop();
        m_socket.close();
        m_ctx.close();
    }

    bool init(const njson &config) override {
        try {
            if (config.contains("endpoint")) {
                m_endpoint = config["endpoint"].get<std::string>();
            }

            if (config.contains("compression")) {
                m_use_compression = config["compression"].get<bool>();
            }

            if (m_use_compression && !m_gpu_encoder) {
                m_gpu_encoder = std::make_unique<Utils::NvJpegEncoder>();
            }

            SPDLOG_INFO("Binding ZMQ Proto Publisher to {} (Compression: {})",
                m_endpoint, m_use_compression ? "ON" : "OFF");

            m_socket.bind(m_endpoint);

            // Drop frames if consumer is slow to prevent memory explosion
            // for one 1080P raw image, it occupies 3 * 1080 * 1920 * 1 / 1024 / 1024 = ~6MB
            m_socket.set(zmq::sockopt::sndhwm, 16);

            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("ZmqProtoPublisher Init Error: {}", e.what());
            return false;
        }
    }

protected:
    void on_frame_ready(cv::cuda::GpuMat &frame, PipelineContext &meta_data) override {
        if (frame.empty()) return;

        try {
            // 1. Prepare the Protobuf Message
            cm::proto::FrameMsg proto_msg;

            // 2. Populate Metadata (Nested Message)
            populate_pipeline_ctx(proto_msg.mutable_ctx(), meta_data);

            // 3. Handle Image Data
            if (m_use_compression) {
                // --- JPEG MODE ---
                if (!m_gpu_encoder) return;

                std::vector<uchar> compressed_data;
                if (m_gpu_encoder->encode(frame, compressed_data, 90)) {
                    proto_msg.set_is_cv_mat(false); // It is a compressed buffer
                    proto_msg.set_frame(compressed_data.data(), compressed_data.size());
                } else {
                    return; // Encode failed
                }
            } else {
                // --- RAW MODE ---
                // Download GPU -> CPU
                frame.download(m_cpu_buffer);

                size_t size_in_bytes = m_cpu_buffer.total() * m_cpu_buffer.elemSize();

                proto_msg.set_is_cv_mat(true); // It is raw pixel data
                proto_msg.set_frame(m_cpu_buffer.data, size_in_bytes);
            }

            // 4. Serialize to String
            std::string serialized_payload;
            if (!proto_msg.SerializeToString(&serialized_payload)) {
                SPDLOG_ERROR("Failed to serialize Protobuf message");
                return;
            }

            // 5. Send over ZMQ
            // We use 'video_proto' as the topic envelope
            // Note: ZMQ messages are multipart. We count payload size primarily.
            m_socket.send(zmq::buffer("video_proto"), zmq::send_flags::sndmore);

            zmq::message_t z_msg(serialized_payload.begin(), serialized_payload.end());
            m_socket.send(z_msg, zmq::send_flags::none);

            // 6. Monitor Egress (Bytes per sec / Frames per sec)
            monitor_egress(serialized_payload.size());

        } catch (const std::exception &e) {
            SPDLOG_ERROR("ZmqProto Send Error: {}", e.what());
        }
    }

private:


    void monitor_egress(size_t payload_size) {
        m_egress_bytes_accum += payload_size;
        m_egress_frames_accum++;

        auto now = std::chrono::steady_clock::now();
        auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(now - m_last_report_time).count();

        if (elapsed_sec >= m_report_interval_sec) {
            double duration = static_cast<double>(elapsed_sec);

            // Calculate MegaBytes per second
            double mb_total = static_cast<double>(m_egress_bytes_accum) / (1024.0 * 1024.0);
            double mbps = mb_total / duration;

            // Calculate FPS
            double fps_out = static_cast<double>(m_egress_frames_accum) / duration;

            SPDLOG_INFO("ZMQ Egress [Last {}s]: Rate: {:.2f} MB/s | Sent: {:.1f} FPS | Total: {:.2f} MB",
                        elapsed_sec, mbps, fps_out, mb_total);

            // Reset counters
            m_egress_bytes_accum = 0;
            m_egress_frames_accum = 0;
            m_last_report_time = now;
        }
    }

    static void populate_pipeline_ctx(cm::proto::ProcessingUnitContext* dest, const PipelineContext& src) {
        if (!dest) return;
        dest->set_captured_from_real_device(src.captured_from_real_device);
        dest->set_capture_timestamp_ms(src.capture_timestamp_ms);
        dest->set_capture_from_this_device_since_ms(src.capture_from_this_device_since_ms);
        dest->set_frame_seq_num(src.frame_seq_num);
        dest->set_processing_unit_idx(src.processing_unit_idx);
        dest->set_change_rate(src.change_rate);
        dest->set_fps(src.fps);
    }
};

} // namespace MatrixPipeline::ProcessingUnit
*/