#include "ipc.h"
#include "global_vars.h"
#include "pc_queue.h"
#include "snapshot.pb.h"
#include "utils.h"

#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/detail/os_file_functions.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/permissions.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <spdlog/spdlog.h>
#include <zmq.hpp>

#include <chrono>
#include <fstream>
#include <regex>
#include <sstream>
#include <sys/mman.h>

// rwx,rw,rw
#define PERMS (S_IRWXU | (S_IRGRP | S_IWGRP) | (S_IROTH | S_IWOTH))
// #define SEM_INITIAL_VALUE 1

using namespace boost::interprocess;
using namespace std;

namespace CudaMotion {
// PIMPL idiom
class IPC::impl {
public:
  string deviceName;
  size_t deviceIndex;
  string jpegBytes;

  // HTTP variables
  bool httpEnabled = false;

  // File variables
  bool fileEnabled = false;
  string filePathWithStaticVarEvaluated;

  // ZeroMQ variables
  bool zmqEnabled = false;
  bool zmqSendCVMat = false;
  string zeroMQEndpoint;
  zmq::context_t zmqContext;
  zmq::socket_t zmqSocket;

  // POSIX/Boost shared memory variables
  shared_memory_object shm;
  mapped_region mapped_reg;
  bool shm_enabled = false;
  string shm_name;
  size_t shm_size;
  // void *memPtr;
  // sem_t *semPtr;
  // int shmFd;
  string sem_name;

  //
  PcQueue<ipcQueueElement, ipcDequeueContext> ipcPcQueue =
      PcQueue<ipcQueueElement, ipcDequeueContext>(&ev_flag, 128);

  impl(const size_t deviceIndex, string deviceName, IPC *parent)
      : zmqContext(1), zmqSocket(zmqContext, zmq::socket_type::pub) {
    this->deviceName = deviceName;
    this->deviceIndex = deviceIndex;
    ipcDequeueContext ctx = {.ipcInstance = parent, .ele = {-1, -1, cv::Mat()}};
    ipcPcQueue.start(ctx);
  }

  ~impl() {
    if (!shm_enabled)
      return;
    /*
  if (sem_unlink(sem_name.c_str()) != 0) {
    spdlog::error(
        "sem_unlink() failed: {}({}), but there is nothing we can do", errno,
        strerror(errno));
  }
  if (sem_close(semPtr) != 0) {
    spdlog::error(
        "sem_close() failed: {}({}), but there is nothing we can do", errno,
        strerror(errno));
  }*/
    shared_memory_object::remove(shm_name.c_str());
    named_semaphore::remove(sem_name.c_str());
  }

  bool isHttpEnabled() { return httpEnabled; }

  void enqueueData(ipcQueueElement &eqpl) {
    // cv::Mat operates with an internal reference-counter, so we need to
    // clone() to increase the counter Another point is that try_enqueue() does
    // std::move() internally, how does it interplay with cv::Mat's ref-counting
    // model? Not 100% clear to me...
    /*  if (q.try_enqueue(dispFrame.clone()) == false) [[unlikely]] {
      spdlog::warn("IPC pcQueue is full, this dispFrame will be not be sent");
      }*/
    if (!ipcPcQueue.try_enqueue(eqpl)) [[unlikely]] {
      spdlog::warn(
          "[{}] IPC pcQueue is full, this dispFrame will be not be sent",
          deviceName);
    }
  }

  void enableHttp() {
    httpEnabled = true;
    spdlog::info("[{}] HTTP IPC enabled, endpoint is " HTTP_IPC_URL
                 "?deviceId={}",
                 deviceName, deviceIndex);
  }

  void enableFile(const string &filePathWithStaticVarEvaluated) {
    fileEnabled = true;
    this->filePathWithStaticVarEvaluated = filePathWithStaticVarEvaluated;
    spdlog::info("[{}] IPC via filesystem enabled, filePath: {}", deviceName,
                 filePathWithStaticVarEvaluated);
  }

  void sendDataCb(ipcQueueElement &eqpl) {
    cv::Mat mat;
    SnapshotMsg msg;
    // vector<int> configs = {IMWRITE_JPEG_QUALITY, 80};
    vector<int> configs = {};
    vector<uchar> jpeg_buffer;
    if (httpEnabled) {
      lock_guard<mutex> guard(mutexLiveImage);
      mat = eqpl.snapshot.clone();
      // As of 2023-11-28, cv::imencode does not appear to have CUDA equivalent
      // in OpenCV
      cv::imencode(".jpg", mat, jpeg_buffer, configs);
      jpegBytes = string((const char *)jpeg_buffer.data(), jpeg_buffer.size());
    } else if (fileEnabled || shm_enabled || zmqEnabled) {
      mat = eqpl.snapshot.clone();
      cv::imencode(".jpg", mat, jpeg_buffer, configs);
      jpegBytes = string((const char *)jpeg_buffer.data(), jpeg_buffer.size());
    }
    if (shm_enabled || zmqEnabled) {
      msg.set_rateofchange(eqpl.rateOfChange);
      msg.set_cooldown(eqpl.cooldown);
      msg.set_unixepochns(chrono::time_point_cast<chrono::nanoseconds>(
                              chrono::system_clock::now())
                              .time_since_epoch()
                              .count());
    }
    // Due to some reason, lock_guard<mutex> guard(mutexLiveImage); only
    // mat = dispFrame.clone(); is not enough, we must also lock cv::imencode()
    // to avoid race condition.
    // cv::imencode(".jpg", mat, encodedJpgImage, configs);

    // Profiling show that the above mutex section without actual
    // waiting takes ~30 ms to complete, meaning that the CPU can only
    // handle ~30 fps

    // https://stackoverflow.com/questions/7054844/is-rename-atomic
    // https://stackoverflow.com/questions/29261648/atomic-writing-to-file-on-linux
    if (fileEnabled) {
      sendDataViaFile();
    }

    if (shm_enabled) {
      sendDataViaSharedMemory(msg);
    }
    if (zmqEnabled) {
      // profiling shows that for 1280x720 images, the method takes 10-5000us to
      // complete auto start = chrono::high_resolution_clock::now();
      sendDataViaZeroMQ(mat, msg);
      // auto end = chrono::high_resolution_clock::now();
      // spdlog::info(
      //     "{}us",
      //    chrono::duration_cast<chrono::microseconds>(end - start).count());
    }
  }

  void enableZeroMQ(const string &zeroMQEndpoint, const bool sendCVMat) {
    zmqEnabled = true;
    zmqSendCVMat = sendCVMat;
    try {
      zmqSocket.bind(zeroMQEndpoint);
      spdlog::info("[{}] ZeroMQ IPC enabled, endpoint is {}", deviceName,
                   zeroMQEndpoint);
    } catch (const std::exception &e) {
      spdlog::error(
          "[{}] zmqSocket.bind(zeroMQEndpoint) failed: {}, zeroMQ IPC "
          "support will be disabled for this device",
          deviceName, e.what());
      zmqEnabled = false;
    }
  }

  void enableSharedMemory(const string &sharedMemoryName,
                          const size_t sharedMemSize,
                          const std::string &semaphoreName) {
    shm_enabled = true;
    this->shm_name = sharedMemoryName;
    this->shm_size = sharedMemSize;
    this->sem_name = semaphoreName;
    try {
      shm = shared_memory_object(create_only, sharedMemoryName.c_str(),
                                 read_write);
      shm.truncate(sharedMemSize);
      mapped_reg = mapped_region(shm, read_write);

      // umask() is needed to set the correct permissions.
      auto old_umask = umask(0);

      named_semaphore::remove(sem_name.c_str());
      permissions perm(PERMS);
      named_semaphore sem(create_only_t(), sem_name.c_str(), 8, perm);

      umask(old_umask);

      /*if (semPtr == SEM_FAILED) {
        spdlog::error("[{}] sem_open() failed, {}({})", deviceName, errno,
                      strerror(errno));
        throw interprocess_exception("dummy");
      }*/
      spdlog::info("[{}] Shared memory IPC enabled, shared memory path: {}, "
                   "semaphore path: {}",
                   deviceName, sharedMemoryName, semaphoreName);
      return;
    } catch (interprocess_exception &ex) {
      shared_memory_object::remove(shm_name.c_str());
      named_semaphore::remove(sem_name.c_str());
      spdlog::error("Failed to enable shared memory IPC: {}", ex.what());
      shm_enabled = false;
    }
  }

  void sendDataViaFile() {
    string sp = regex_replace(filePathWithStaticVarEvaluated,
                              regex(R"(\{\{timestamp\}\})"),
                              Utils::getCurrentTimestamp());
    ofstream fout(sp + ".tmp", ios::out | ios::binary);
    if (!fout.good())
      spdlog::error("[{}] Failed to open file [{}]: {}", deviceName,
                    sp + ".tmp", strerror(errno));
    return;

    fout.write(jpegBytes.data(), jpegBytes.size());
    if (!fout.good()) {
      spdlog::error("[{}] Failed to write to file [{}]: {}", deviceName,
                    sp + ".tmp", strerror(errno));
    }
    fout.close();
    if (rename((sp + ".tmp").c_str(), sp.c_str()) != 0) {
      spdlog::error("[{}] Failed to rename [{}] tp [{}]: {}", deviceName,
                    sp + ".tmp", sp, strerror(errno));
    }
    // profiling shows from ofstream fout()... to rename() takes
    // less than 1 ms.}
  }

  void sendDataViaSharedMemory(SnapshotMsg msg) {
    msg.clear_cvmatbytes();
    msg.set_jpegbytes(jpegBytes);
    size_t s = msg.ByteSizeLong();
    if (s > shm_size - sizeof(size_t)) {
      spdlog::error("[{}] encodedJpgImage({} bytes) too large for "
                    "sharedMemSize({} bytes)",
                    deviceName, s, shm_size);
      s = 0;
    } else {
      try {
        named_semaphore sem(open_only_t(), sem_name.c_str());
        sem.wait();
        memcpy(mapped_reg.get_address(), &s, sizeof(size_t));
        memcpy((uint8_t *)mapped_reg.get_address() + sizeof(size_t),
               msg.SerializeAsString().data(), s);
        sem.post();
      } catch (interprocess_exception &ex) {
        spdlog::error("named_semaphore exception: {}", ex.what());
      }
    }
  }

  void sendDataViaZeroMQ(cv::Mat mat, SnapshotMsg msg) {
    // sending images with ProtoBuf over ZeroMQ is not cheap, profiling shows
    // transfering one 1280x720 image takes time that ranges from a few to a
    // few hundred milliseconds
    if (zmqSendCVMat) {
      msg.clear_jpegbytes();
      msg.set_cvmatbytes(mat.data, mat.total() * mat.elemSize());
    } else {
      msg.clear_cvmatbytes();
      msg.set_jpegbytes(jpegBytes);
    }
    auto serializedMsg = msg.SerializeAsString();

    try {
      if (auto ret =
              zmqSocket.send(
                  zmq::const_buffer(serializedMsg.data(), serializedMsg.size()),
                  zmq::send_flags::none) != serializedMsg.size()) {
        spdlog::error("zmqSocket.send() failed: ZeroMQ socket reports {} bytes "
                      "being sent, but serializedMsg.size() is {} bytes",
                      ret, serializedMsg.size());
      }
    } catch (const zmq::error_t &err) {
      spdlog::error("zmqSocket.send() failed: {}({}). The program will "
                    "continue with this frame being unsent",
                    err.num(), err.what());
    }
  }

  void wait() { ipcPcQueue.wait(); }
};

IPC::IPC(const size_t deviceIndex, const string &deviceName)
    : pimpl{std::make_unique<impl>(deviceIndex, deviceName, this)} {}

void IPC::enableZeroMQ(const string &zeroMQEndpoint, const bool sendCVMat) {
  pimpl->enableZeroMQ(zeroMQEndpoint, sendCVMat);
}

void IPC::enableHttp() { pimpl->enableHttp(); }

void IPC::enableFile(const string &filePathWithStaticVarEvaluated) {
  pimpl->enableFile(filePathWithStaticVarEvaluated);
}

bool IPC::isHttpEnabled() { return pimpl->isHttpEnabled(); }

IPC::~IPC() {}

void IPC::wait() { pimpl->wait(); }

void IPC::enqueueData(ipcQueueElement &eqpl) { pimpl->enqueueData(eqpl); }

void IPC::sendDataCb(ipcQueueElement &eqpl) { pimpl->sendDataCb(eqpl); }

void IPC::enableSharedMemory(const string &sharedMemoryName,
                             const size_t sharedMemSize,
                             const std::string &semaphoreName) {
  pimpl->enableSharedMemory(sharedMemoryName, sharedMemSize, semaphoreName);
}

string IPC::getJpegBytes() { return pimpl->jpegBytes; }
} // namespace CudaMotion