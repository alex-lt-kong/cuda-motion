
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
typedef oatpp::String String;

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

    // clang-format off
    String html = "<html lang='en'>"
                  "  <head>"
                  "    <title>" PROJECT_NAME "'s HTTP service</title>"
                  "    <meta charset=utf-8/>"
                  "  </head>"
                  "  <body>"
                  "    <p>Hi " + authObject->User + ", welcome to " PROJECT_NAME "'s HTTP service</p>"
                  "    <p>Try accessing real-time images like below:</p>" +
                        settings.value("/httpService/advertisedAddress"_json_pointer,
                        "http://localhost:54321") + HTTP_IPC_URL "?deviceId=0"
                  "  </body>"
                  "</html>";
    // clang-format on
    auto response = createResponse(Status::CODE_200, html);
    response->putHeader(Header::CONTENT_TYPE, "text/html");
    return response;
  }

  ENDPOINT("GET", HTTP_IPC_URL "*", live_image,
           REQUEST(std::shared_ptr<IncomingRequest>, request),
           AUTHORIZATION(std::shared_ptr<MyAuthorizationObject>, authObject)) {
    // Just to silence the "unused object warning..."
    (void)authObject;
    /* get url 'tail' - everything that comes after '*' */
    String tail = request->getPathTail(); // everything that goes after '*'
    /* check tail for null */
    OATPP_ASSERT_HTTP(tail, Status::CODE_400, "null query-params");
    auto queryParams = oatpp::network::Url::Parser::parseQueryParams(tail);

    /* get your param by name */
    auto queryParameter = queryParams.get("deviceId");
    size_t deviceId = 0;
    if (queryParameter != nullptr) {
      deviceId = atoi(queryParameter->c_str());
      if (deviceId < myDevices.size()) {
        std::vector<uint8_t> encodedImg;
        myDevices[deviceId]->getLiveImage(encodedImg);
        if (encodedImg.size() > 0) {
          String buf =
              String((const char *)encodedImg.data(), encodedImg.size());
          return Response::createShared(Status::CODE_200,
                                        BufferBody::createShared(buf));
        } else {
          return Response::createShared(
              Status::CODE_200,
              BufferBody::createShared(
                  "ERROR: video device exists but fetched image is empty. The "
                  "most likely reason is that http snapshot for the given "
                  "device is turned off"));
        }
      } else {
        return Response::createShared(
            Status::CODE_400,
            BufferBody::createShared("ERROR: Invalid deviceId"));
      }
    }
    return Response::createShared(
        Status::CODE_400,
        BufferBody::createShared("ERROR: Invalid parameters"));
  }
};

#include OATPP_CODEGEN_END(ApiController) //<- End Codegen

#endif /* CS_STATIC_CONTROLLER_HPP */
