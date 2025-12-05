#pragma once

#include "../interfaces/i_asynchronous_processing_unit.h"
#include "../utils.h"
#include "frame_msg.pb.h" 

#include <zmq.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <nvjpeg.h> 
#include <spdlog/spdlog.h>

#include <string>
#include <vector>
#include <memory>

namespace CudaMotion::ProcessingUnit {
/**
 * @brief Publishes frames + metadata using Google Protocol Buffers over ZeroMQ.
 */
class ZmqPublisher : public IAsynchronousProcessingUnit {
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

public:
    ZmqPublisher() : m_socket(m_ctx, zmq::socket_type::pub) {}

    ~ZmqPublisher() override {
        stop();
        m_socket.close();
        m_ctx.close();
    }

    /**
     * @brief Configures ZMQ Endpoint and Compression mode.
     * JSON: { "endpoint": "tcp://*:5555", "compression": true }
     */
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
            m_socket.set(zmq::sockopt::sndhwm, 5);

            return true;
        } catch (const std::exception &e) {
            SPDLOG_ERROR("ZmqProtoPublisher Init Error: {}", e.what());
            return false;
        }
    }

protected:
    void on_frame_ready(cv::cuda::GpuMat &frame, ProcessingMetaData &meta_data) override {
        if (frame.empty()) return;

        try {
            // 1. Prepare the Protobuf Message
            cm::proto::FrameMsg proto_msg;

            // 2. Populate Metadata (Nested Message)
            populate_metadata(proto_msg.mutable_meta_data(), meta_data);

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
                
                // Note: The Proto definition provided lacks width/height/type fields.
                // The receiver must know the dimensions implicitly or via side-channel 
                // if 'is_cv_mat' is true.
            }

            // 4. Serialize to String
            std::string serialized_payload;
            if (!proto_msg.SerializeToString(&serialized_payload)) {
                SPDLOG_ERROR("Failed to serialize Protobuf message");
                return;
            }

            // 5. Send over ZMQ
            // We use 'video_proto' as the topic envelope
            m_socket.send(zmq::buffer("video_proto"), zmq::send_flags::sndmore);
            
            zmq::message_t z_msg(serialized_payload.begin(), serialized_payload.end());
            m_socket.send(z_msg, zmq::send_flags::none);

        } catch (const std::exception &e) {
            SPDLOG_ERROR("ZmqProto Send Error: {}", e.what());
        }
    }

private:
    void populate_metadata(cm::proto::ProcessingMetaData* dest, const ProcessingMetaData& src) {
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

} // namespace CudaMotion::ProcessingUnit