#include "device_manager.h"
#include "global_vars.h"
#include "http_service/oatpp_entry.h"
#include "utils.h"

#include <crow.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

using namespace std;
using json = nlohmann::json;

void askForCred(crow::response &res) {
  res.set_header("WWW-Authenticate", "Basic realm=On-demand CCTV server");
  res.code = 401;
  res.write("<h1>Unauthorized access</h1>");
  res.end();
}

string httpAuthenticate(const crow::request &req) {
  string myauth = req.get_header_value("Authorization");
  if (myauth.size() < 6) {
    return "";
  }
  string mycreds = myauth.substr(6);
  string d_mycreds = crow::utility::base64decode(mycreds, mycreds.size());
  size_t found = d_mycreds.find(':');
  string username = d_mycreds.substr(0, found);
  string password = d_mycreds.substr(found + 1);
  if (!settings["httpService"]["httpAuthentication"]["accounts"].contains(
          username)) {
    return "";
  }
  if (settings["httpService"]["httpAuthentication"]["accounts"][username] !=
      password) {
    return "";
  }
  return username;
}

struct httpAuthMiddleware : crow::ILocalMiddleware {
  struct context {};

  void before_handle(crow::request &req, crow::response &res,
                     __attribute__((unused)) context &ctx) {
    if (settings["httpService"]["httpAuthentication"]["enabled"]) {
      string username = httpAuthenticate(req);
      if (username.size() == 0) {
        askForCred(res);
        return;
      }
    }
  }

  void after_handle(__attribute__((unused)) crow::request &req,
                    __attribute__((unused)) crow::response &res,
                    __attribute__((unused)) context &ctx) {}
};

crow::App<httpAuthMiddleware> app;
std::vector<deviceManager *> myDevices;

static void signal_handler(int signum) {
  if (signum != SIGCHLD) {
    ev_flag = 1;
    /* Internally, Crow appears to be using io_context. Is io_context.stop()
    reentrant then? The document does not directly answer this:
    https://www.boost.org/doc/libs/1_76_0/doc/html/boost_asio/reference/io_context/stop.html
    but the wording appears to imply yes. */
    app.stop();
  }
  char msg[] = "Signal [  ] caught\n";
  msg[8] = '0' + signum / 10;
  msg[9] = '0' + signum % 10;
  size_t len = sizeof(msg) - 1;
  size_t written = 0;
  while (written < len) {
    ssize_t ret = write(STDOUT_FILENO, msg + written, len - written);
    if (ret == -1) {
      perror("write()");
      break;
    }
    written += ret;
  }
}

void install_signal_handler() {
  static_assert(_NSIG < 99,
                "signal_handler() can't handle more than 99 signals");

  struct sigaction act;
  // Initialize the signal set to empty, similar to memset(0)
  if (sigemptyset(&act.sa_mask) == -1) {
    perror("sigemptyset()");
    abort();
  }
  act.sa_handler = signal_handler;
  /* SA_RESETHAND means we want our signal_handler() to intercept the signal
  once. If a signal is sent twice, the default signal handler will be used
  again. `man sigaction` describes more possible sa_flags. */
  /* In this particular case, we should not enable SA_RESETHAND, mainly
  due to the issue that if a child process is kill, multiple SIGPIPE will
  be invoked consecutively, breaking the program.  */
  // act.sa_flags = SA_RESETHAND;
  if (sigaction(SIGINT, &act, 0) + sigaction(SIGABRT, &act, 0) +
          sigaction(SIGQUIT, &act, 0) + sigaction(SIGTERM, &act, 0) +
          sigaction(SIGPIPE, &act, 0) + sigaction(SIGCHLD, &act, 0) +
          sigaction(SIGTRAP, &act, 0) <
      0) {
    throw runtime_error("sigaction() called failed: " + to_string(errno) + "(" +
                        strerror(errno) + ")");
  }
}

class CustomLogger : public crow::ILogHandler {
public:
  CustomLogger() {}
  void log(string message, crow::LogLevel level) {
    if (level <= crow::LogLevel::INFO) {
      spdlog::info("CrowCpp: {}", message);
    } else if (level < crow::LogLevel::WARNING) {
      spdlog::warn("CrowCpp: {}", message);
    } else {
      spdlog::error("CrowCpp: {}", message);
    }
  }
};

void start_http_server() {

  CustomLogger logger;
  crow::logger::setHandler(&logger);
  app.loglevel(crow::LogLevel::Info);

  CROW_ROUTE(app, "/").CROW_MIDDLEWARES(app, httpAuthMiddleware)([]() {
    return string("HTTP service running\nTry: ") +
           (settings["httpService"]["ssl"]["enabled"] ? "https" : "http") +
           "://<host>:" + to_string(settings["httpService"]["port"]) +
           "/live_image/?deviceId=0";
  });

  CROW_ROUTE(app, "/live_image/")
      .CROW_MIDDLEWARES(app, httpAuthMiddleware)([](const crow::request &req,
                                                    crow::response &res) {
        if (req.url_params.get("deviceId") == nullptr) {
          res.code = 400;
          res.set_header("Content-Type", "application/json");
          res.end(crow::json::wvalue(
                      {{"status", "error"}, {"data", "deviceId not specified"}})
                      .dump());
          return;
        }
        uint32_t deviceId = atoi(req.url_params.get("deviceId"));
        if (deviceId > myDevices.size() - 1) {
          res.code = 400;
          res.set_header("Content-Type", "application/json");
          res.end(
              crow::json::wvalue(
                  {{"status", "error"},
                   {"data", "deviceId " + to_string(deviceId) + " is invalid"}})
                  .dump());
          return;
        }
        vector<uint8_t> encodedImg;
        myDevices[deviceId]->getLiveImage(encodedImg);
        if (encodedImg.size() > 0) {
          res.set_header("Content-Type", "image/jpg");
          res.end(string((char *)(encodedImg.data()), encodedImg.size()));
        } else {
          res.code = 404;
          res.set_header("Content-Type", "application/json");
          res.end(crow::json::wvalue(
                      {{"status", "error"},
                       {"data", "Image data not found for deviceId " +
                                    to_string(deviceId) +
                                    ". Perhaps http snapshot is disabled?"}})
                      .dump());
        }
      });

  if (settings["httpService"]["ssl"]["enabled"]) {
    app.bindaddr(settings["httpService"]["interface"])
        .ssl_file(settings["httpService"]["ssl"]["crtPath"],
                  settings["httpService"]["ssl"]["keyPath"])
        .port(settings["httpService"]["port"])
        .signal_clear()
        .multithreaded()
        .run();
  } else {
    app.bindaddr(settings["httpService"]["interface"])
        .port(settings["httpService"]["port"])
        .signal_clear()
        .multithreaded()
        .run();
  }
}

int main(int argc, char *argv[]) {
  string settingsPath;
  if (argc > 2) {
    cerr << "Usage: " << argv[0] << " [config-file.jsonc]" << endl;
    return EXIT_FAILURE;
  } else if (argc == 2) {
    settingsPath = string(argv[1]);
  } else {
    settingsPath =
        string(getenv("HOME")) + "/.config/ak-studio/camera-server.jsonc";
  }
  // Doc: https://github.com/gabime/spdlog/wiki/3.-Custom-formatting
  spdlog::set_pattern("%Y-%m-%dT%T.%e%z|%5t|%8l| %v");
  spdlog::info("Camera Server started");
  install_signal_handler();

  spdlog::info("cv::getBuildInformation(): {}", string(getBuildInformation()));

  spdlog::info("Loading json settings from {}", settingsPath);
  ifstream is(settingsPath);
  settings = json::parse(is,
                         /* callback */ nullptr,
                         /* allow exceptions */ true,
                         /* ignore_comments */ true);

  size_t deviceCount = settings["devices"].size();
  if (deviceCount == 0) {
    throw logic_error("No devices are defined.");
  }
  myDevices = vector<deviceManager *>(deviceCount);
  // myDevices.reserve(deviceCount);
  for (size_t i = 0; i < deviceCount; ++i) {
    /* Seems ZeroMQ objects are neither copyable nor movable, using
    pointer is a relatively easy way to circumvent these operations. */
    myDevices[i] = new deviceManager(i, settings["devicesDefault"],
                                     settings["devices"][i]);
    myDevices[i]->StartInternalEventLoopThread();
  }
  initialize_http_service(settings["httpService"]["interface"].get<string>(),
                          settings["httpService"]["port"].get<int>() % 10000);
  start_http_server();
  for (size_t i = 0; i < myDevices.size(); ++i) {
    myDevices[i]->WaitForInternalEventLoopThreadToExit();
    spdlog::info("{}-th device thread exited gracefully", i);
    delete myDevices[i];
  }
  spdlog::info("All device threads exited gracefully");

  return 0;
}
