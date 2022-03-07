#pragma once
#include <string>

using namespace std;

class logger {

public:
  logger(string logPath, bool enableDebug);
  void setLogPath(string logPath);
  void debug(string content);
  void info(string content);
  void error(string content);
  
  string prependTimestamp();

private:
  string logPath = "";
  bool enableDebug = false;

  void writeToFile(string content);
};
