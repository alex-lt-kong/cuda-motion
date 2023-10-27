#ifndef CS_APP_COMPONENT_HPP
#define CS_APP_COMPONENT_HPP

#include "../global_vars.h"
#include "error_handler.hpp"

#include <oatpp-openssl/Config.hpp>
#include <oatpp-openssl/server/ConnectionProvider.hpp>
#include <oatpp/core/macro/component.hpp>
#include <oatpp/network/tcp/server/ConnectionProvider.hpp>
#include <oatpp/parser/json/mapping/ObjectMapper.hpp>
#include <oatpp/web/server/HttpConnectionHandler.hpp>
#include <oatpp/web/server/HttpRouter.hpp>

namespace on = oatpp::network;
typedef on::ServerConnectionProvider ServerConnectionProvider;
typedef on::tcp::server::ConnectionProvider ConnectionProvider;
typedef on::ConnectionHandler ConnectionHandler;
typedef oatpp::web::server::HttpRouter HttpRouter;
typedef oatpp::web::server::HttpConnectionHandler HttpConnectionHandler;

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
  OATPP_CREATE_COMPONENT(std::shared_ptr<ServerConnectionProvider>,
                         serverConnectionProvider)
  ([this]() -> std::shared_ptr<ServerConnectionProvider> {
    std::shared_ptr<oatpp::openssl::Config> config = nullptr;
    if (settings["httpService"]["ssl"]["enabled"].get<bool>()) {
      auto config = oatpp::openssl::Config::createDefaultServerConfigShared(
          settings["httpService"]["ssl"]["pemPath"].get<std::string>().c_str(),
          settings["httpService"]["ssl"]["keyPath"].get<std::string>().c_str());
      return oatpp::openssl::server::ConnectionProvider::createShared(
          config, {host, port});
    }
    return on::tcp::server::ConnectionProvider::createShared(
        {host, port, on::Address::IP_4});
  }());

  /**
   *  Create Router component
   */
  OATPP_CREATE_COMPONENT(std::shared_ptr<HttpRouter>, httpRouter)
  ([] { return HttpRouter::createShared(); }());

  /**
   *  Create ConnectionHandler component which uses Router component to route
   * requests
   */
  OATPP_CREATE_COMPONENT(std::shared_ptr<ConnectionHandler>,
                         serverConnectionHandler)
  ([] {
    OATPP_COMPONENT(std::shared_ptr<HttpRouter>,
                    router); // get Router component
    OATPP_COMPONENT(std::shared_ptr<oatpp::data::mapping::ObjectMapper>,
                    objectMapper); // get ObjectMapper component

    auto connectionHandler = HttpConnectionHandler::createShared(router);
    connectionHandler->setErrorHandler(
        std::make_shared<ErrorHandler>(objectMapper));
    return connectionHandler;
  }());
};

#endif /* CS_APP_COMPONENT_HPP */
