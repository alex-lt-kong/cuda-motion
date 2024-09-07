#ifndef CM_IPC_H
#define CM_IPC_H

//#pragma GCC diagnostic push
//#pragma GCC diagnostic ignored "-Wdeprecated-enum-enum-conversion"
#include <opencv2/core/core.hpp>
//#pragma GCC diagnostic pop

namespace CudaMotion {
class IPC;

struct ipcQueueElement {
  float rateOfChange;
  int64_t cooldown;
  cv::Mat snapshot;
};

struct ipcDequeueContext {
  IPC *ipcInstance;
  ipcQueueElement ele;
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
  void enqueueData(ipcQueueElement &eqpl);
  std::vector<uint8_t> getEncodedJpgImage();
  void sendDataCb(ipcQueueElement &eqpl);
  bool isHttpEnabled();
  void wait();

private:
  class impl;
  std::unique_ptr<impl> pimpl;
};
} // namespace CudaMotion
#endif // CM_IPC_H
