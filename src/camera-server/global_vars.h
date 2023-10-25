#ifndef CS_GLOBAL_VARS_H
#define CS_GLOBAL_VARS_H

#include <mutex>
#include <signal.h>

/* The extern keyword tells the compiler that please dont generate any
definition for it when compiling the source files that include the header.
Without extern, multiple object files that include this header file
will generate its own version of ev_flag, causing the "multiple definition
of `ev_flag';" error. By adding extern, we need to manually add the definition
of ev_flag in one .c/.cpp file. In this particular case, this is done
in main.cpp. */
extern volatile sig_atomic_t ev_flag;

enum MotionDetectionMode {
  MODE_ALWAYS_RECORD = 2,
  MODE_DETECT_MOTION = 1,
  MODE_DISABLED = 0
};

extern std::mutex mutexLiveImage;

#endif /* CS_GLOBAL_VARS_H */
