#pragma once

namespace MatrixPipeline::ProcessingUnit {
enum SynchronousProcessingResult {
  success_and_continue = 0,
  success_and_stop = 1,
  failure_and_continue = 2,
  failure_and_stop = 3
};
}