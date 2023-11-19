#ifndef CS_IPC_H
#define CS_IPC_H

#include "readerwriterqueue/readerwritercircularbuffer.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-anon-enum-enum-conversion"
#pragma GCC diagnostic ignored "-Wc11-extensions"
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <zmq.hpp>
#pragma GCC diagnostic pop

#include <semaphore.h>
#include <thread>

#define PERMS (S_IRWXU | S_IRWXG | S_IRWXO)
#define SEM_INITIAL_VALUE 1

class IPC {
public:
  IPC(const size_t deviceIndex, const std::string &deviceName);
  void enableZeroMQ(const std::string &zeroMQEndpoint);
  void enableSharedMemory(const std::string &sharedMemoryName,
                          size_t sharedMemSize,
                          const std::string &semaphoreName);
  void enableHttp();
  void enableFile(const std::string &filePathWithStaticVarEvaluated);
  ~IPC();
  void enqueueData(cv::Mat dispFrame);
  std::vector<uint8_t> encodedJpgImage;
  std::thread consumer;
  static void consume(IPC *);
  void consumeCb(cv::Mat &disspFrame);
  inline bool isHttpEnabled() { return httpEnabled; }

private:
  moodycamel::BlockingReaderWriterCircularBuffer<cv::Mat> q;
  size_t deviceIndex;
  std::string deviceName = "<unset>";
  // File variables
  bool fileEnabled = false;
  std::string filePathWithStaticVarEvaluated;

  // HTTP variables
  bool httpEnabled = false;

  // ZeroMQ variables
  bool zmqEnabled = false;
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

  void sendDataViaZeroMQ();
  void sendDataViaSharedMemory();
};

#endif // CS_IPC_H
