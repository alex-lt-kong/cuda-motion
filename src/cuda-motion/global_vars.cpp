#include "global_vars.h"

volatile sig_atomic_t ev_flag = 0;

std::mutex mutexLiveImage;
std::mutex mtxNjsonSettings;

nlohmann::json settings;
