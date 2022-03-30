#include "logger.h"
#include <chrono>
#include <fstream> 

using namespace std;
using sysclock_t = std::chrono::system_clock;

logger::logger(string logPath, bool enableDebug) {
  this->logPath = logPath;
  this->enableDebug = enableDebug;
}

void logger::setLogPath(string logPath) {
  this->logPath = logPath;
}

string logger::prependTimestamp(){
    time_t now = sysclock_t::to_time_t(sysclock_t::now());
    //"19700101_000000"
    char buf[16] = { 0 };
    std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", std::localtime(&now));
    // https://www.cplusplus.com/reference/ctime/strftime/
    return std::string(buf);
}

void logger::debug(string content) {
  if (this->enableDebug == false) { return; }
  this->writeToFile(this->prependTimestamp() + " | DEBUG | " + content + "\n");
}

void logger::error(string content) {
  this->writeToFile(this->prependTimestamp() + " | ERROR | " + content + "\n");
}


void logger::info(string content) {
  this->writeToFile(this->prependTimestamp() + " | INFO  | " + content + "\n");
}

void logger::writeToFile(string content) {
  ofstream myfile;
  myfile.open(logPath, std::ofstream::out | std::ofstream::app);
  myfile << content;
  myfile.close();
}