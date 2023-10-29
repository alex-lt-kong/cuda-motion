#ifndef CS_IPC_H
#define CS_IPC_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <zmq.hpp>

#include <semaphore.h>

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
  void sendData(cv::Mat &dispFrame);
  std::vector<uint8_t> encodedJpgImage;

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
};

#endif /* CS_IPC_H */
