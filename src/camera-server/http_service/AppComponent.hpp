#ifndef CS_APP_COMPONENT_HPP
#define CS_APP_COMPONENT_HPP

#include "error_handler.hpp"

#include <oatpp/core/macro/component.hpp>
#include <oatpp/network/tcp/server/ConnectionProvider.hpp>
#include <oatpp/parser/json/mapping/ObjectMapper.hpp>
#include <oatpp/web/server/HttpConnectionHandler.hpp>
#include <oatpp/web/server/HttpRouter.hpp>

#include <iostream>

namespace on = oatpp::network;
namespace ows = oatpp::web::server;

/**
 *  Class which creates and holds Application components and registers
 * components in oatpp::base::Environment Order of components initialization is
 * from top to bottom
 */
class AppComponent {
private:
  oatpp::String host;
  v_uint16 port;

public:
  AppComponent(oatpp::String host, v_uint16 port) : host(host), port(port) {}

  /**
   * Create ObjectMapper component to serialize/deserialize DTOs in Controller's
   * API
   */
  OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::data::mapping::ObjectMapper>,
                         apiObjectMapper)
  ([] {
    auto objectMapper =
        oatpp::parser::json::mapping::ObjectMapper::createShared();
    objectMapper->getDeserializer()->getConfig()->allowUnknownFields = false;
    return objectMapper;
  }());

  /**
   *  Create ConnectionProvider component which listens on the port
   */
  OATPP_CREATE_COMPONENT(std::shared_ptr<on::ServerConnectionProvider>,
                         serverConnectionProvider)
  ([this] {
    return on::tcp::server::ConnectionProvider::createShared(
        {host, port, on::Address::IP_4});
  }());

  /**
   *  Create Router component
   */
  OATPP_CREATE_COMPONENT(std::shared_ptr<ows::HttpRouter>, httpRouter)
  ([] { return ows::HttpRouter::createShared(); }());

  /**
   *  Create ConnectionHandler component which uses Router component to route
   * requests
   */
  OATPP_CREATE_COMPONENT(std::shared_ptr<on::ConnectionHandler>,
                         serverConnectionHandler)
  ([] {
    OATPP_COMPONENT(std::shared_ptr<ows::HttpRouter>,
                    router); // get Router component
    OATPP_COMPONENT(std::shared_ptr<oatpp::data::mapping::ObjectMapper>,
                    objectMapper); // get ObjectMapper component

    auto connectionHandler = ows::HttpConnectionHandler::createShared(router);
    connectionHandler->setErrorHandler(
        std::make_shared<ErrorHandler>(objectMapper));
    return connectionHandler;
  }());
};

#endif /* CS_APP_COMPONENT_HPP */
