#include "cuda_helper.h"

#include <boost/stacktrace.hpp>
#include <spdlog/spdlog.h>

namespace MatrixPipeline::Utils {

void TRTLogger::log(Severity severity, const char *msg) noexcept {
  if (severity <= Severity::kWARNING) {
    SPDLOG_LOGGER_CALL(
        spdlog::default_logger(),
        (severity == Severity::kERROR ? spdlog::level::err
                                      : spdlog::level::warn),
        "[TensorRT] {}\n{}", msg,
        boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
  }
}

TRTLogger g_logger; // The actual memory allocation happens here

} // namespace MatrixPipeline::Utils