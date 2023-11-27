#ifndef CM_AUTH_HANDLER_H
#define CM_AUTH_HANDLER_H

#include "../global_vars.h"

#include <memory>
#include <mycrypto/misc.h>
#include <mycrypto/misc.hpp>
#include <mycrypto/sha256.h>
#include <oatpp/core/Types.hpp>
#include <oatpp/core/base/Environment.hpp>
#include <oatpp/web/server/handler/AuthorizationHandler.hpp>
#include <spdlog/spdlog.h>
#include <string.h>

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
    unsigned char sha256_bytes[SHA256_HASH_SIZE];
    if (cal_sha256_hash(
            reinterpret_cast<const unsigned char *>(password->c_str()),
            strlen(password->c_str()), sha256_bytes) != 0) {
      spdlog::warn("cal_sha256_hash() failed");
      return nullptr;
    }
    auto sha256_char = unique_fptr<char>(
        bytes_to_hex_string(sha256_bytes, SHA256_HASH_SIZE, false));
    if (sha256_char == nullptr) {
      spdlog::warn("bytes_to_hex_string() failed");
      return nullptr;
    }

    if (strcmp(settings["httpService"]["httpAuthentication"][userId->c_str()]
                   .get<std::string>()
                   .c_str(),
               sha256_char.get()) != 0) {
      spdlog::warn("password is incorrect for user [{}]", password->c_str());
      return nullptr;
    }
    auto authObject = MyAuthorizationObject::createShared();
    authObject->User = userId;
    return authObject;
  }
};

#endif /* CM_AUTH_HANDLER_H */
