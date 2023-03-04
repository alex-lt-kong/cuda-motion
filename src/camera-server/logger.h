#pragma once
#include <string>

using namespace std;

class logger {

public:
  logger(string logPath, bool enableDebug);
  void setLogPath(string logPath);
  void debug(string moduleName, string content);
  void info(string moduleName, string content);
  void error(string moduleName, string content);
  
  string prependTimestamp();

private:
  string logPath = "";
  bool enableDebug = false;
  int moduleNameLength = 20;

  void writeToFile(string content);
};
