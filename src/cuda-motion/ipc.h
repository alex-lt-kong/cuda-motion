#ifndef CM_IPC_H
#define CM_IPC_H

#include "pc_queue.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-enum-enum-conversion"
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <zmq.hpp>
#pragma GCC diagnostic pop
#include <readerwriterqueue/readerwritercircularbuffer.h>

#include <semaphore.h>
#include <thread>

#define PERMS (S_IRWXU | S_IRWXG | S_IRWXO)
#define SEM_INITIAL_VALUE 1

class IPC;

struct ipcQueuePayload {
  IPC *ipcInstance;
  cv::Mat snapshot;
};

class IPC {
public:
  IPC(const size_t deviceIndex, const std::string &deviceName);
  void enableZeroMQ(const std::string &zeroMQEndpoint, const bool sendCVMat);
  void enableSharedMemory(const std::string &sharedMemoryName,
                          size_t sharedMemSize,
                          const std::string &semaphoreName);
  void enableHttp();
  void enableFile(const std::string &filePathWithStaticVarEvaluated);
  ~IPC();
  void enqueueData(cv::Mat dispFrame);
  std::vector<uint8_t> encodedJpgImage;
  cv::Mat mat;
  void sendDataCb(cv::Mat &dispFrame);
  inline bool isHttpEnabled() { return httpEnabled; }
  void wait();

private:
  size_t deviceIndex;
  std::string deviceName = "<unset>";
  // File variables
  bool fileEnabled = false;
  std::string filePathWithStaticVarEvaluated;

  // HTTP variables
  bool httpEnabled = false;

  // ZeroMQ variables
  bool zmqEnabled = false;
  bool zmqSendCVMat = false;
  std::string zeroMQEndpoint;
  zmq::context_t zmqContext;
  zmq::socket_t zmqSocket;

  // POSIX shared memory variables
  bool sharedMemEnabled = false;
  std::string sharedMemoryName;
  size_t sharedMemSize;
  void *memPtr;
  sem_t *semPtr;
  int shmFd;
  std::string semaphoreName;

  PcQueue<cv::Mat, ipcQueuePayload, ipcQueuePayload> ipcPcQueue;
  void sendDataViaZeroMQ();
  void sendDataViaSharedMemory();
  void sendDataViaFile();
};

#endif // CM_IPC_H
