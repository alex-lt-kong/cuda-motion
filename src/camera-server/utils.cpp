#include <fcntl.h>
#include <iomanip>
#include <poll.h>
#include <sstream>
#include <sys/wait.h>
#include <time.h>

#include <spdlog/spdlog.h>

#include "deviceManager.h"
#include "utils.h"

void exec(void *This, const vector<string> &args, string &stdoutStr,
          string &stderrStr, int &rc) {

  struct timespec start, now;
  long elapsed_usecs = 0;
  long timeout_usecs = 3600 * 1000000L; // 3600 sec
  vector<char *> cargs;
  cargs.reserve(args.size() + 1);

  for (size_t i = 0; i < args.size(); ++i) {
    cargs.push_back(const_cast<char *>(args[i].c_str()));
  }
  cargs.push_back(NULL);

  int pipefd_out[2], pipefd_err[2];
  // pipefd[0] is the read end of the pipe
  // pipefd[1] is the write end of the pipe
  char buffer[4096];

  if (pipe(pipefd_out) == -1) {
    spdlog::error("[{}] pipe(pipefd_out) failed, {}({})",
                  reinterpret_cast<deviceManager *>(This)->getDeviceName(),
                  errno, strerror(errno));
  }
  if (pipe(pipefd_err) == -1) {
    spdlog::error("[{}] pipe(pipefd_err) failed, {}({})",
                  reinterpret_cast<deviceManager *>(This)->getDeviceName(),
                  errno, strerror(errno));
    if (close(pipefd_out[0]) == -1 || close(pipefd_out[1] == -1)) {
      spdlog::error("[{}] close(): {}({})",
                    reinterpret_cast<deviceManager *>(This)->getDeviceName(),
                    errno, strerror(errno));
    }
  }

  pid_t child_pid = fork(); // span a child process

  if (child_pid == -1) { // fork() failed, no child process created
    spdlog::error("[{}] fork() failed, {}({})",
                  reinterpret_cast<deviceManager *>(This)->getDeviceName(),
                  errno, strerror(errno));
    if (close(pipefd_err[0]) == -1 || close(pipefd_err[1]) == -1 ||
        close(pipefd_out[0]) == -1 || close(pipefd_out[1]) == -1) {
      spdlog::error("[{}] close(): {}({})",
                    reinterpret_cast<deviceManager *>(This)->getDeviceName(),
                    errno, strerror(errno));
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &start);
  if (child_pid == 0) { // fork() succeeded, we are in the child process
    if (close(pipefd_out[0]) == -1) {
      spdlog::error("[{}] close(pipefd_stdout[0]) in child failed: {}({})",
                    reinterpret_cast<deviceManager *>(This)->getDeviceName(),
                    errno, strerror(errno));
    }
    if (close(pipefd_err[0]) == -1) {
      spdlog::error("[{}] close(pipefd_err[0]) in child failed: {}({})",
                    reinterpret_cast<deviceManager *>(This)->getDeviceName(),
                    errno, strerror(errno));
    }
    // man dup2
    if (dup2(pipefd_out[1], STDOUT_FILENO) == -1 ||
        dup2(pipefd_err[1], STDERR_FILENO) == -1) {
      spdlog::error("[{}] dup2() in child failed, "
                    "some output may be missing: {}({})",
                    reinterpret_cast<deviceManager *>(This)->getDeviceName(),
                    errno, strerror(errno));
    }

    execv(cargs[0], cargs.data());
    spdlog::error("[{}] execv() in child failed, {}({})",
                  reinterpret_cast<deviceManager *>(This)->getDeviceName(),
                  errno, strerror(errno));
    // The exec() functions return only if an error has occurred.
    // The return value is -1, and errno is set to indicate the error.
    return;
  }
  spdlog::info("[{}] child process fork()ed successfully",
               reinterpret_cast<deviceManager *>(This)->getDeviceName());
  // Only parent gets here
  if (close(pipefd_out[1]) == -1 || close(pipefd_err[1]) == -1) {
    spdlog::error("[{}] close() in parent failed: {}({})",
                  reinterpret_cast<deviceManager *>(This)->getDeviceName(),
                  errno, strerror(errno));
  }

  struct pollfd pfds[] = {
      {pipefd_out[0], POLLIN, 0},
      {pipefd_err[0], POLLIN, 0},
  };
  int nfds = sizeof(pfds) / sizeof(struct pollfd);
  int open_fds = nfds;

  while (open_fds > 0) {
    if (elapsed_usecs >= timeout_usecs) {
      spdlog::error("[{}] Command execution timed out after {} seconds",
                    reinterpret_cast<deviceManager *>(This)->getDeviceName(),
                    timeout_usecs / 1000000);
      break;
    }
    clock_gettime(CLOCK_MONOTONIC, &now);
    elapsed_usecs = (now.tv_sec - start.tv_sec) * 1000000 +
                    (now.tv_nsec - start.tv_nsec) / 1000;
    if (poll(pfds, nfds, 5000) == -1) {
      spdlog::error("[{}] poll() failed, some output may be missing: "
                    "{}({})",
                    reinterpret_cast<deviceManager *>(This)->getDeviceName(),
                    errno, strerror(errno));
    }

    for (int j = 0; j < nfds; j++) {
      if (pfds[j].revents == 0) {
        continue;
      }
      if (pfds[j].revents & POLLIN) {
        memset(buffer, 0, sizeof buffer);
        ssize_t s = read(pfds[j].fd, buffer, sizeof buffer - 1);
        if (s == -1) {
          spdlog::error(
              "[{}] read() failed, some output may be "
              "missing: {}({})",
              reinterpret_cast<deviceManager *>(This)->getDeviceName(), errno,
              strerror(errno));
        }
        if (j == 0) {
          stdoutStr += buffer;
        } else {
          stderrStr += buffer;
        }
      } else { /* POLLERR | POLLHUP */
        open_fds--;
      }
    }
  }

  if (close(pipefd_out[0]) == -1 || close(pipefd_err[0]) == -1) {
    spdlog::error("[{}] close() in parent failed: {}({})",
                  reinterpret_cast<deviceManager *>(This)->getDeviceName(),
                  errno, strerror(errno));
  }

  int status;
  __useconds_t sleep_us = 1;
  while (waitpid(child_pid, &status, WNOHANG) == 0) {
    usleep(sleep_us);
    sleep_us = sleep_us >= 1000000 ? sleep_us : sleep_us * 2;
    clock_gettime(CLOCK_MONOTONIC, &now);
    elapsed_usecs = (now.tv_sec - start.tv_sec) * 1000000 +
                    (now.tv_nsec - start.tv_nsec) / 1000;

    if (elapsed_usecs > timeout_usecs) {
      spdlog::warn("[{}] Timeout {} ms reached, kill()ing process {}...",
                   reinterpret_cast<deviceManager *>(This)->getDeviceName(),
                   timeout_usecs / 1000, child_pid);
      // This avoid leaving a zombie process in the process table:
      // https://stackoverflow.com/questions/69509427/kill-child-process-spawned-with-execl-without-making-it-zombie
      signal(SIGCHLD, SIG_IGN);
      if (kill(child_pid, SIGTERM) == -1) {
        spdlog::error("[{}] kill(SIGTERM) failed, {}({})",
                      reinterpret_cast<deviceManager *>(This)->getDeviceName(),
                      errno, strerror(errno));
        if (kill(child_pid, SIGKILL) == -1) {
          spdlog::error(
              "[{}] kill(SIGKILL) failed, {}({})",
              reinterpret_cast<deviceManager *>(This)->getDeviceName(), errno,
              strerror(errno));
        } else {
          spdlog::info(
              "[{}] kill()ed successfully with SIGKILL",
              reinterpret_cast<deviceManager *>(This)->getDeviceName());
          break;
        }
      } else {
        spdlog::info("[{}] kill()ed successfully with SIGTERM",
                     reinterpret_cast<deviceManager *>(This)->getDeviceName());
        break; // Without break the while() loop could be entered again.
      }
    }
  }
  if (WIFEXITED(status)) {
    rc = WEXITSTATUS(status);
  } else {
    rc = EXIT_FAILURE;
    if (WIFSIGNALED(status)) {
      spdlog::error("[{}] child process exited abnormally "
                    "(terminated by a signal: {})",
                    reinterpret_cast<deviceManager *>(This)->getDeviceName(),
                    WTERMSIG(status));
    } else if (WIFSTOPPED(status)) {
      spdlog::error("[{}] child process exited abnormally "
                    "(stopped by delivery of a signal: {})",
                    reinterpret_cast<deviceManager *>(This)->getDeviceName(),
                    WSTOPSIG(status));
    } else {
      spdlog::error("[{}] child process exited abnormally "
                    "(unknown status: {})",
                    reinterpret_cast<deviceManager *>(This)->getDeviceName(),
                    status);
    }
  }
  return;
}

void execAsync(void *This, const vector<string> &args, exec_cb cb) {
  auto f = [](void *This, const vector<string> &args, exec_cb cb) {
    string stdout, stderr;
    int rc;
    exec(This, args, stdout, stderr, rc);
    cb(This, stdout, stderr, rc);
  };
  thread th_exec(f, This, args, cb);
  th_exec.detach();
}
