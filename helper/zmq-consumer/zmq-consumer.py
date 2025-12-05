import zmq
import frame_msg_pb2 # Generated from: protoc --python_out=. frame_msg.proto
import cv2
import numpy as np
import struct
import os

# --- Configuration ---
ZMQ_ENDPOINT = os.environ.get("ZMQ_ENDPOINT", "tcp://localhost:5555")
TOPIC = "video_proto"

# RAW MODE SETTINGS
# Since the Proto message doesn't carry dimensions, we must know them a priori
# if receiving raw pixel data (is_cv_mat = true).
# Ignored if receiving JPEG (is_cv_mat = false).
RAW_WIDTH = 540
RAW_HEIGHT = 336
RAW_CHANNELS = 3

def main():
    # 1. Setup ZMQ Subscriber
    context = zmq.Context()
    socket = context.socket(zmq.SUB)

    print(f"Connecting to {ZMQ_ENDPOINT}...")
    socket.connect(ZMQ_ENDPOINT)

    # Subscribe to the topic envelope
    socket.setsockopt_string(zmq.SUBSCRIBE, TOPIC)

    print(f"Waiting for frames on topic '{TOPIC}'...")

    while True:
        try:
            # 2. Receive Multipart Message: [Topic, Protobuf Payload]
            # recv_multipart returns a list of bytes
            parts = socket.recv_multipart()
            if len(parts) < 2:
                continue

            topic = parts[0]
            payload = parts[1]

            # 3. Deserialize Protobuf
            msg = frame_msg_pb2.FrameMsg()
            msg.ParseFromString(payload)

            # 4. Access ctx
            ctx = msg.ctx

            # Print stats to console
            print(f"Seq: {ctx.frame_seq_num} | "
                  f"FPS: {ctx.fps:.1f} | "
                  f"Change: {ctx.change_rate:.4f} | "
                  f"Mode: {'RAW' if msg.is_cv_mat else 'JPEG'}")

            # 5. Decode Image
            frame_bytes = msg.frame
            img = None

            if not msg.is_cv_mat:
                # --- JPEG Mode ---
                # Buffer is a complete encoded image file. OpenCV handles it automatically.
                np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                # --- RAW Mode ---
                # Buffer is raw pixels. We must reshape manually.
                np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)

                expected_size = RAW_WIDTH * RAW_HEIGHT * RAW_CHANNELS
                if len(np_arr) != expected_size:
                    print(f"Error: Raw data size {len(np_arr)} does not match config {RAW_WIDTH}x{RAW_HEIGHT}")
                    continue

                img = np_arr.reshape((RAW_HEIGHT, RAW_WIDTH, RAW_CHANNELS))

            # 6. Display
            if img is not None:
                # Optional: Overlay metadata on the frame for visual confirmation
                #cv2.putText(img, f"FPS: {ctx.fps:.1f}", (10, 30),
                # cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("ZMQ Protobuf Stream", img)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            print("Stopping...")
            break
        except Exception as e:
            print(f"Error: {e}")

    socket.close()
    context.term()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()