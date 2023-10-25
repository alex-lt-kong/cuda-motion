#include "global_vars.h"

volatile sig_atomic_t ev_flag = 0;

std::mutex mutexLiveImage;

nlohmann::json settings;
