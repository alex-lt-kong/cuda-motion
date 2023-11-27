
#include "error_handler.hpp"

ErrorHandler::ErrorHandler(
    const std::shared_ptr<oatpp::data::mapping::ObjectMapper> &objectMapper)
    : m_objectMapper(objectMapper) {}

std::shared_ptr<ErrorHandler::OutgoingResponse>
ErrorHandler::handleError(const Status &status, const oatpp::String &message,
                          const Headers &headers) {

  auto response =
      ResponseFactory::createResponse(status, message, m_objectMapper);

  for (const auto &pair : headers.getAll()) {
    response->putHeader(pair.first.toString(), pair.second.toString());
  }

  return response;
}