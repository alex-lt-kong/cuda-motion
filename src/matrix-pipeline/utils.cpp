#include "utils.h"

#include <spdlog/spdlog.h>

#include <atomic>
#include <signal.h>
#include <thread>
#include <time.h>

using namespace std;

namespace MatrixPipeline::Utils {

signal_handler_callback sh_callback;

atomic<ssize_t> executionCounter = -1;

static void signal_handler(int signum) noexcept {
  if (signum == SIGCHLD) {
    // When a child process stops or terminates, SIGCHLD is sent to the parent
    // process. The default response to the signal is to ignore it.
    return;
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
  sh_callback(signum);
}

void install_signal_handler(signal_handler_callback cb) {
  static_assert(_NSIG < 99,
                "signal_handler() can't handle more than 99 signals");

  sh_callback = cb;
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

} // namespace MatrixPipeline::Utils