#include "ipc.h"
#include "global_vars.h"
#include "snapshot.pb.h"
#include "utils.h"

#include <chrono>
#include <spdlog/spdlog.h>

#include <fstream>
#include <regex>
#include <sstream>
#include <sys/mman.h>
#include <zmq.hpp>

using namespace std;

IPC::IPC(const size_t deviceIndex, const string &deviceName)
    : zmqContext(1), zmqSocket(zmqContext, zmq::socket_type::pub) {
  this->deviceIndex = deviceIndex;
  this->deviceName = deviceName;
  ipcPcQueue.start({.ipcInstance = this, .snapshot = cv::Mat()});
}

void IPC::enableZeroMQ(const string &zeroMQEndpoint) {
  zmqEnabled = true;
  try {
    zmqSocket.bind(zeroMQEndpoint);
    spdlog::info("[{}] ZeroMQ IPC enabled, endpoint is {}", deviceName,
                 zeroMQEndpoint);
  } catch (const std::exception &e) {
    spdlog::error("[{}] zmqSocket.bind(zeroMQEndpoint) failed: {}, zeroMQ IPC "
                  "support will be disabled for this device",
                  deviceName, e.what());
    zmqEnabled = false;
  }
}

void IPC::enableHttp() {
  httpEnabled = true;
  spdlog::info("[{}] HTTP IPC enabled, endpoint is {}" HTTP_IPC_URL
               "?deviceId={}",
               deviceName,
               settings.value("/httpService/advertisedAddress"_json_pointer,
                              "http://localhost:54321"),
               deviceIndex);
}

void IPC::enableFile(const string &filePathWithStaticVarEvaluated) {
  fileEnabled = true;
  this->filePathWithStaticVarEvaluated = filePathWithStaticVarEvaluated;
  spdlog::info("[{}] IPC via filesystem enabled, filePath: {}", deviceName,
               filePathWithStaticVarEvaluated);
}

void IPC::enableSharedMemory(const string &sharedMemoryName,
                             const size_t sharedMemSize,
                             const std::string &semaphoreName) {
  sharedMemEnabled = true;
  this->sharedMemoryName = sharedMemoryName;
  this->sharedMemSize = sharedMemSize;
  this->semaphoreName = semaphoreName;

  // umask() is needed to set the correct permissions.
  mode_t old_umask = umask(0);
  // Should have used multiple RAII classes to handle this but...
  shmFd = shm_open(sharedMemoryName.c_str(), O_RDWR | O_CREAT, PERMS);
  if (shmFd < 0) {
    spdlog::error(
        "[{}] shm_open() failed, {}({}), shared memory will be disabled "
        "for this device",
        deviceName, errno, strerror(errno));
    goto err_shmFd;
  }
  if (ftruncate(shmFd, sharedMemSize) != 0) {
    spdlog::error(
        "[{}] ftruncate() failed, {}({}), shared memory will be disabled "
        "for this device",
        deviceName, errno, strerror(errno));
    goto err_ftruncate;
  }
  memPtr =
      mmap(NULL, sharedMemSize, PROT_READ | PROT_WRITE, MAP_SHARED, shmFd, 0);
  if (memPtr == MAP_FAILED) {
    spdlog::error("[{}] mmap() failed, {}({}) shared memory will be disabled "
                  "for this device",
                  deviceName, errno, strerror(errno));
    goto err_mmap;
  }
  semPtr = sem_open(semaphoreName.c_str(), O_CREAT | O_RDWR, PERMS,
                    SEM_INITIAL_VALUE);
  umask(old_umask);

  if (semPtr == SEM_FAILED) {
    spdlog::error("[{}] sem_open() failed, {}({})", deviceName, errno,
                  strerror(errno));
    goto err_sem_open;
  }
  spdlog::info("[{}] Shared memory IPC enabled, shared memory path: {}, "
               "semaphore path: {}",
               deviceName, sharedMemoryName, semaphoreName);
  return;
err_sem_open:
  if (munmap(memPtr, sharedMemSize) != 0) {
    spdlog::error(
        "[{}] munmap() failed: {}({}), but there is nothing we can do",
        deviceName, errno, strerror(errno));
  }
err_mmap:
err_ftruncate:
  if (shm_unlink(sharedMemoryName.c_str()) != 0)
    spdlog::error(
        "[{}] shm_unlink() failed: {}({}), but there is nothing we can do",
        deviceName, errno, strerror(errno));
  if (close(shmFd) != 0)
    spdlog::error("[{}] close() failed: {}({}), but there is nothing we can do",
                  deviceName, errno, strerror(errno));
err_shmFd:
  sharedMemEnabled = false;
}

IPC::~IPC() {

  if (!sharedMemEnabled)
    return;
  // unlink and close() are both needed: unlink only disassociates the
  // name from the underlying semaphore object, but the semaphore object
  // is not gone. It will only be gone when we close() it.
  if (sem_unlink(semaphoreName.c_str()) != 0) {
    spdlog::error("sem_unlink() failed: {}({}), but there is nothing we can do",
                  errno, strerror(errno));
  }
  if (sem_close(semPtr) != 0) {
    spdlog::error("sem_close() failed: {}({}), but there is nothing we can do",
                  errno, strerror(errno));
  }
  if (munmap(memPtr, sharedMemSize) != 0) {
    spdlog::error("munmap() failed: {}({}), but there is nothing we can do",
                  errno, strerror(errno));
  }
  if (shm_unlink(sharedMemoryName.c_str()) != 0) {
    spdlog::error("shm_unlink() failed: {}({}), but there is nothing we can do",
                  errno, strerror(errno));
  }
  if (close(shmFd) != 0) {
    spdlog::error("close() failed: {}({}), "
                  "but there is nothing we can do",
                  errno, strerror(errno));
  }
}

void IPC::wait() { ipcPcQueue.wait(); }

void IPC::enqueueData(cv::Mat dispFrame) {
  // cv::Mat operates with an internal reference-counter, so we need to clone()
  // to increase the counter
  // Another point is that try_enqueue() does std::move() internally, how does
  // it interplay with cv::Mat's ref-counting model? Not 100% clear to me...
  /*  if (q.try_enqueue(dispFrame.clone()) == false) [[unlikely]] {
    spdlog::warn("IPC pcQueue is full, this dispFrame will be not be sent");
    }*/
  if (!ipcPcQueue.try_enqueue(dispFrame)) [[unlikely]] {
    spdlog::warn("[{}] IPC pcQueue is full, this dispFrame will be not be sent",
                 deviceName);
  }
}

void IPC::sendDataCb(cv::Mat &dispFrame) {
  // vector<int> configs = {IMWRITE_JPEG_QUALITY, 80};
  vector<int> configs = {};
  if (httpEnabled) {
    lock_guard<mutex> guard(mutexLiveImage);
    cv::imencode(".jpg", dispFrame, encodedJpgImage, configs);
  } else if (fileEnabled || sharedMemEnabled || zmqEnabled) {
    cv::imencode(".jpg", dispFrame, encodedJpgImage, configs);
  }
  /*
    {
      auto now = std::chrono::system_clock::now();
      auto local_now = std::chrono::system_clock::to_time_t(now);
      auto iso_time = put_time(localtime(&local_now), "%Y%m%dT%H%M%S.%f");
      ostringstream oss;
      oss << "/tmp/test_" << iso_time << ".jpg";
      ofstream output_file(oss.str(), ios::binary);
      output_file.write(reinterpret_cast<const char *>(encodedJpgImage.data()),
                        encodedJpgImage.size());
      output_file.close();
    }*/

  // Profiling show that the above mutex section without actual
  // waiting takes ~30 ms to complete, means that the CPU can only
  // handle ~30 fps

  // https://stackoverflow.com/questions/7054844/is-rename-atomic
  // https://stackoverflow.com/questions/29261648/atomic-writing-to-file-on-linux
  if (fileEnabled) {
    string sp =
        regex_replace(filePathWithStaticVarEvaluated,
                      regex(R"(\{\{timestamp\}\})"), getCurrentTimestamp());
    ofstream fout(sp + ".tmp", ios::out | ios::binary);
    if (!fout.good()) {
      spdlog::error("[{}] Failed to open file [{}]: {}", deviceName,
                    sp + ".tmp", strerror(errno));
    } else {
      fout.write((char *)encodedJpgImage.data(), encodedJpgImage.size());
      if (!fout.good()) {
        spdlog::error("[{}] Failed to write to file [{}]: {}", deviceName,
                      sp + ".tmp", strerror(errno));
      }
      fout.close();
      if (rename((sp + ".tmp").c_str(), sp.c_str()) != 0) {
        spdlog::error("[{}] Failed to rename [{}] tp [{}]: {}", deviceName,
                      sp + ".tmp", sp, strerror(errno));
      }
    }
    // profiling shows from ofstream fout()... to rename() takes
    // less than 1 ms.
  }

  if (sharedMemEnabled) {
    sendDataViaSharedMemory();
  }
  if (zmqEnabled) {
    // profiling shows that for 1280x720 images, the method takes 10-5000us to
    // complete auto start = chrono::high_resolution_clock::now();
    sendDataViaZeroMQ();
    // auto end = chrono::high_resolution_clock::now();
    // spdlog::info(
    //     "{}us",
    //    chrono::duration_cast<chrono::microseconds>(end - start).count());
  }
}

void IPC::sendDataViaSharedMemory() {
  size_t s = encodedJpgImage.size();
  if (s > sharedMemSize - sizeof(size_t)) {
    spdlog::error("[{}] encodedJpgImage({} bytes) too large for "
                  "sharedMemSize({} bytes)",
                  deviceName, s, sharedMemSize);
    s = 0;
  } else {
    if (sem_wait(semPtr) != 0) {
      spdlog::error(
          "[{}] sem_wait() failed: {}, the program will "
          "continue to run but the semaphore mechanism could be broken",
          deviceName, errno);
    } else {

      memcpy(memPtr, &s, sizeof(size_t));
      memcpy((uint8_t *)memPtr + sizeof(size_t), encodedJpgImage.data(),
             encodedJpgImage.size());
      if (sem_post(semPtr) != 0) {
        spdlog::error(
            "[{}] sem_post() failed: {}, the program will "
            "continue to run but the semaphore mechanism could be broken",
            deviceName, errno);
      }
    }
  }
}

void IPC::sendDataViaZeroMQ() {
  // sending images with ProtoBuf over ZeroMQ is not cheap, profiling shows
  // transfering one 1280x720 image takes time that ranges from a few to a few
  // hundred milliseconds
  SnapshotMsg msg;
  auto epochNanoseconds =
      chrono::time_point_cast<chrono::nanoseconds>(chrono::system_clock::now())
          .time_since_epoch()
          .count();
  msg.set_unixepochns(epochNanoseconds);
  msg.set_payload(encodedJpgImage.data(), encodedJpgImage.size());
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
