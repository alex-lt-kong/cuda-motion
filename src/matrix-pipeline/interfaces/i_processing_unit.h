#pragma once

#include "../entities/processing_context.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace MatrixPipeline::ProcessingUnit {

using njson = nlohmann::json;

class IProcessingUnit {
protected:
  const std::string m_unit_path;
  bool m_is_disabled{false};

public:
  explicit IProcessingUnit(std::string unit_path)
      : m_unit_path(std::move(unit_path)) {
    SPDLOG_INFO("Initializing processing_unit: {}", m_unit_path);
  };
  IProcessingUnit() = default;

  virtual ~IProcessingUnit() {
    SPDLOG_INFO("processing_unit {} destructed", m_unit_path);
  }

  virtual bool init(const njson &config) = 0;

  [[nodiscard]] bool is_disabled() const { return m_is_disabled; };

  /// Disable this unit
  void disable() { m_is_disabled = true; };
};

} // namespace MatrixPipeline::ProcessingUnit