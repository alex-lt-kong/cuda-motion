
#ifndef CS_STATI_CCONTROLLER_HPP
#define CS_STATI_CCONTROLLER_HPP

#include "../auth_handler.h"

#include <oatpp/core/macro/codegen.hpp>
#include <oatpp/core/macro/component.hpp>
#include <oatpp/parser/json/mapping/ObjectMapper.hpp>
#include <oatpp/web/server/api/ApiController.hpp>

#include <iostream>

#include OATPP_CODEGEN_BEGIN(ApiController) //<- Begin Codegen

namespace ows = oatpp::web::server;

class StaticController : public ows::api::ApiController {
public:
  StaticController(const std::shared_ptr<ObjectMapper> &objectMapper)
      : ows::api::ApiController(objectMapper) {
    setDefaultAuthorizationHandler(std::make_shared<MyAuthorizationHandler>());
  }

public:
  static std::shared_ptr<StaticController> createShared(OATPP_COMPONENT(
      std::shared_ptr<ObjectMapper>,
      objectMapper) // Inject objectMapper component here as default parameter
  ) {
    return std::make_shared<StaticController>(objectMapper);
  }

  ENDPOINT("GET", "/", root,
           AUTHORIZATION(std::shared_ptr<MyAuthorizationObject>, authObject)) {

    std::cout << authObject->User->c_str() << std::endl;

    oatpp::String html = "<html lang='en'>"
                         "  <head>"
                         "    <meta charset=utf-8/>"
                         "  </head>"
                         "  <body>"
                         "    <p>" PROJECT_NAME "</p>"
                         "    <a href='swagger/ui'>Checkout Swagger-UI page</a>"
                         "  </body>"
                         "</html>";
    html->append(authObject->User);
    auto response = createResponse(Status::CODE_200, html);
    response->putHeader(Header::CONTENT_TYPE, "text/html");
    return response;
  }
};

#include OATPP_CODEGEN_END(ApiController) //<- End Codegen

#endif /* CS_STATI_CCONTROLLER_HPP */
