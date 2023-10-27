#include "ipc.h"
#include "global_vars.h"
#include "utils.h"

#include <spdlog/spdlog.h>

#include <fstream>
#include <regex>
#include <sys/mman.h>

using namespace std;

IPC::IPC() : zmqContext(1), zmqSocket(zmqContext, zmq::socket_type::pub) {}

void IPC::enableZeroMQ(const string &zeroMQEndpoint) {
  zmqEnabled = true;
  try {
    zmqSocket.bind(zeroMQEndpoint);
  } catch (const std::exception &e) {
    spdlog::error("zmqSocket.bind(zeroMQEndpoint) failed: {}, zeroMQ IPC "
                  "support will be disabled for this device",
                  e.what());
    zmqEnabled = false;
  }
}

void IPC::enableHttp() { httpEnabled = true; }

void IPC::enableFile(const string &filePathWithStaticVarEvaluated) {
  this->filePathWithStaticVarEvaluated = filePathWithStaticVarEvaluated;
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
    spdlog::error("shm_open() failed, {}({}), shared memory will be disabled "
                  "for this device",
                  errno, strerror(errno));
    goto err_shmFd;
  }
  if (ftruncate(shmFd, sharedMemSize) != 0) {
    spdlog::error("ftruncate() failed, {}({}), shared memory will be disabled "
                  "for this device",
                  errno, strerror(errno));
    goto err_ftruncate;
  }
  memPtr =
      mmap(NULL, sharedMemSize, PROT_READ | PROT_WRITE, MAP_SHARED, shmFd, 0);
  if (memPtr == MAP_FAILED) {
    spdlog::error("mmap() failed, {}({}) shared memory will be disabled "
                  "for this device",
                  errno, strerror(errno));
    goto err_mmap;
  }
  semPtr = sem_open(semaphoreName.c_str(), O_CREAT | O_RDWR, PERMS,
                    SEM_INITIAL_VALUE);
  umask(old_umask);

  if (semPtr == SEM_FAILED) {
    spdlog::error("sem_open() failed, {}({})", errno, strerror(errno));
    goto err_sem_open;
  }

  return;
err_sem_open:
  if (munmap(memPtr, sharedMemSize) != 0) {
    spdlog::error("munmap() failed: {}({}), "
                  "but there is nothing we can do",
                  errno, strerror(errno));
  }
err_mmap:
err_ftruncate:
  if (shm_unlink(sharedMemoryName.c_str()) != 0)
    spdlog::error("shm_unlink() failed: {}({}), "
                  "but there is nothing we can do",
                  errno, strerror(errno));
  if (close(shmFd) != 0)
    spdlog::error("close() failed: {}({}), "
                  "but there is nothing we can do",
                  errno, strerror(errno));
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
    spdlog::error("sem_unlink() failed: {}({}), "
                  "but there is nothing we can do",
                  errno, strerror(errno));
  }
  if (sem_close(semPtr) != 0) {
    spdlog::error("sem_close() failed: {}({}), "
                  "but there is nothing we can do",
                  errno, strerror(errno));
  }
  if (munmap(memPtr, sharedMemSize) != 0) {
    spdlog::error("munmap() failed: {}({}), "
                  "but there is nothing we can do",
                  errno, strerror(errno));
  }
  if (shm_unlink(sharedMemoryName.c_str()) != 0) {
    spdlog::error("shm_unlink() failed: {}({}), "
                  "but there is nothing we can do",
                  errno, strerror(errno));
  }
  if (close(shmFd) != 0) {
    spdlog::error("close() failed: {}({}), "
                  "but there is nothing we can do",
                  errno, strerror(errno));
  }
}

void IPC::sendData(cv::Mat &dispFrame) {
  // vector<int> configs = {IMWRITE_JPEG_QUALITY, 80};
  vector<int> configs = {};
  if (httpEnabled) {
    lock_guard<mutex> guard(mutexLiveImage);
    cv::imencode(".jpg", dispFrame, encodedJpgImage, configs);
  } else {
    cv::imencode(".jpg", dispFrame, encodedJpgImage, configs);
  }

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

  if (zmqEnabled) {
    zmqSocket.send(encodedJpgImage.data(), encodedJpgImage.size());
  }
}
