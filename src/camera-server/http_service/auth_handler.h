#ifndef CS_AUTH_HANDLER_HPP
#define CS_AUTH_HANDLER_HPP

#include <oatpp/core/Types.hpp>
#include <oatpp/core/base/Environment.hpp>
#include <oatpp/web/server/handler/AuthorizationHandler.hpp>

#include <iostream>

namespace owsh = oatpp::web::server::handler;

class MyAuthorizationObject : public owsh::AuthorizationObject {
public:
  oatpp::String User;

  static std::shared_ptr<MyAuthorizationObject> createShared() {
    return std::make_shared<MyAuthorizationObject>();
  }
};

class MyAuthorizationHandler : public owsh::BasicAuthorizationHandler {
public:
  MyAuthorizationHandler()
      : owsh::BasicAuthorizationHandler(PROJECT_NAME "-realm") {}

  // override `authorize` function with your own logic
  // `authorize` gets called with the token received in the bearer-authorization
  // header.
  std::shared_ptr<AuthorizationObject>
  authorize(const oatpp::String &userId, const oatpp::String &password) {
    if (userId == "foo" && password == "bar") {
      auto authObject = MyAuthorizationObject::createShared();
      authObject->User = "foo";
      return authObject;
    }
    return nullptr;
  }
};

#endif /* CS_AUTH_HANDLER_HPP */
