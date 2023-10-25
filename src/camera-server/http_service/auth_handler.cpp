#include <oatpp/core/Types.hpp>
#include <oatpp/core/base/Environment.hpp>
#include <oatpp/web/server/handler/AuthorizationHandler.hpp>

class MyAuthorizationObject
    : public oatpp::web::server::handler::AuthorizationObject {
public:
  oatpp::String User;
  v_uint64 Id;
  oatpp::String Type;
};

class MyAuthorizationHandler
    : public oatpp::web::server::handler::BearerAuthorizationHandler {
public:
  MyAuthorizationHandler()
      : oatpp::web::server::handler::BearerAuthorizationHandler(
            "custom-bearer-realm") // Set realm in parent-constructor
  {}

  // override `authorize` function with your own logic
  // `authorize` gets called with the token received in the bearer-authorization
  // header.
  std::shared_ptr<AuthorizationObject>
  authorize(const oatpp::String &token) override {

    auto authObject = MyAuthorizationObject::createShared();
    authObject->User = myDatabaseResult->User;
    authObject->Id = myDatabaseResult->Id;
    authObject->Type = myDatabaseResult->Type;
    return authObject;

    // In case the token is invalid or the authorization fails, return nullptr
    // and Oat++ will reject the request and will not call your endpoint
    return nullptr;
  }
};