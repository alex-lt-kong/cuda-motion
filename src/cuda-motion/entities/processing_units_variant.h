#pragma once

#include "../interfaces/i_asynchronous_processing_unit.h"
#include "../interfaces/i_synchronous_processing_unit.h"

#include <variant>

namespace CudaMotion::ProcessingUnit {
// class IAsynchronousProcessingUnit;

using ProcessingUnitVariant =
    std::variant<std::unique_ptr<ISynchronousProcessingUnit>,
                 std::unique_ptr<IAsynchronousProcessingUnit>>;

// --- C++17/C++20 Overload Helper ---
template <class... Ts> struct overload : Ts... {
  using Ts::operator()...;
};
template <class... Ts> overload(Ts...) -> overload<Ts...>;

} // namespace CudaMotion::ProcessingUnit