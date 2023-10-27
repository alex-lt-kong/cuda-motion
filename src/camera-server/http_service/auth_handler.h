#ifndef CS_AUTH_HANDLER_H
#define CS_AUTH_HANDLER_H

#include "../global_vars.h"

#include <oatpp/core/Types.hpp>
#include <oatpp/core/base/Environment.hpp>
#include <oatpp/web/server/handler/AuthorizationHandler.hpp>
#include <spdlog/spdlog.h>

#include <iostream>

typedef oatpp::String oString;
typedef oatpp::web::server::handler::BasicAuthorizationHandler
    BasicAuthorizationHandler;
typedef oatpp::web::server::handler::AuthorizationObject AuthorizationObject;

class MyAuthorizationObject : public AuthorizationObject {
public:
  oString User;

  static std::shared_ptr<MyAuthorizationObject> createShared() {
    return std::make_shared<MyAuthorizationObject>();
  }
};

class MyAuthorizationHandler : public BasicAuthorizationHandler {
public:
  MyAuthorizationHandler() : BasicAuthorizationHandler(PROJECT_NAME "-realm") {}

  // override `authorize` function with your own logic
  // `authorize` gets called with the token received in the bearer-authorization
  // header.
  std::shared_ptr<AuthorizationObject> authorize(const oString &userId,
                                                 const oString &password) {
    if (!settings["httpService"]["httpAuthentication"].contains(
            userId->c_str())) {
      spdlog::warn("User [{}] does not exist", userId->c_str());
      return nullptr;
    }
    if (settings["httpService"]["httpAuthentication"][userId->c_str()] !=
        password->c_str()) {
      spdlog::warn("password is incorrect for user [{}]", password->c_str());
      return nullptr;
    }
    auto authObject = MyAuthorizationObject::createShared();
    authObject->User = userId;
    return authObject;
  }
};

#endif /* CS_AUTH_HANDLER_H */
