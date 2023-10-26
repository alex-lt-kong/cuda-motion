
#ifndef CS_STATIC_CONTROLLER_HPP
#define CS_STATIC_CONTROLLER_HPP

#include "../../device_manager.h"
#include "../auth_handler.h"

#include <cstdlib>
#include <oatpp/core/Types.hpp>
#include <oatpp/core/macro/codegen.hpp>
#include <oatpp/core/macro/component.hpp>
#include <oatpp/parser/json/mapping/ObjectMapper.hpp>
#include <oatpp/web/protocol/http/outgoing/BufferBody.hpp>
#include <oatpp/web/protocol/http/outgoing/Response.hpp>
#include <oatpp/web/server/api/ApiController.hpp>

#include OATPP_CODEGEN_BEGIN(ApiController) //<- Begin Codegen

typedef oatpp::web::protocol::http::outgoing::BufferBody BufferBody;
typedef oatpp::web::protocol::http::outgoing::Response Response;
typedef oatpp::web::server::api::ApiController ApiController;

class StaticController : public ApiController {
public:
  StaticController(const std::shared_ptr<ObjectMapper> &objectMapper)
      : ApiController(objectMapper) {
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

  ENDPOINT("GET", "/live_image/*", live_image,
           REQUEST(std::shared_ptr<IncomingRequest>, request),
           AUTHORIZATION(std::shared_ptr<MyAuthorizationObject>, authObject)) {

    /* get url 'tail' - everything that comes after '*' */
    String tail = request->getPathTail(); // everything that goes after '*'

    /* check tail for null */
    OATPP_ASSERT_HTTP(tail, Status::CODE_400, "null query-params");

    /* parse query params from tail */
    auto queryParams = oatpp::network::Url::Parser::parseQueryParams(tail);

    /* get your param by name */
    auto queryParameter = queryParams.get("deviceId");
    uint32_t deviceId = 0;
    if (queryParameter != nullptr) {
      spdlog::info("{}", queryParameter->c_str());
      deviceId = atoi(queryParameter->c_str());
    }
    if (deviceId < myDevices.size()) {
      std::vector<uint8_t> encodedImg;
      myDevices[deviceId]->getLiveImage(encodedImg);
      oatpp::String buf =
          oatpp::String((const char *)encodedImg.data(), encodedImg.size());
      auto body = BufferBody::createShared(buf);
      return Response::createShared(Status::CODE_200, body);
    } else {
      return Response::createShared(Status::CODE_200,
                                    BufferBody::createShared("ERROR"));
    }
  }
};

#include OATPP_CODEGEN_END(ApiController) //<- End Codegen

#endif /* CS_STATIC_CONTROLLER_HPP */
