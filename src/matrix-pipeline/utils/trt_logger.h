#pragma once

#include <NvInfer.h>

namespace MatrixPipeline::Utils {

class TRTLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override;
};

// Use 'extern' to tell other modules this exists elsewhere
extern TRTLogger g_logger;

} // namespace MatrixPipeline::Utils