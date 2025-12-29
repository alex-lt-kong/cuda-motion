#include "cuda_helper.h"

#include <spdlog/spdlog.h>
namespace MatrixPipeline::Utils {

void TRTLogger::log(Severity severity, const char *msg) noexcept {
  if (severity <= Severity::kWARNING) {
    SPDLOG_LOGGER_CALL(spdlog::default_logger(),
                       (severity == Severity::kERROR ? spdlog::level::err
                                                     : spdlog::level::warn),
                       "[TensorRT] {}", msg);
  }
}

TRTLogger g_logger; // The actual memory allocation happens here

} // namespace MatrixPipeline::Utils