#include <sys/wait.h>
#include <iomanip>
#include <sstream>
#include <poll.h>
#include <fcntl.h>

#include <spdlog/spdlog.h>

#include "utils.h"

void exec(const vector<string>& args, string& stdoutStr, string& stderrStr, int& rc) {

    vector<char*> cargs;
    cargs.reserve(args.size() + 1);

    for(size_t i = 0; i < args.size(); ++i) {
        cargs.push_back(const_cast<char*>(args[i].c_str()));
    }
    cargs.push_back(NULL);

    int pipefd_out[2], pipefd_err[2];
    // pipefd[0] is the read end of the pipe
    // pipefd[1] is the write end of the pipe
    char buffer[4096];

    if (pipe(pipefd_out) == -1) {      
        spdlog::error("pipe(pipefd_out) failed, errno: {}", errno);
    }
    if (pipe(pipefd_err) == -1) {
        spdlog::error("pipe(pipefd_err) failed, errno: {}", errno);
        if (close(pipefd_out[0]) == -1 || close(pipefd_out[1] == -1)) {
            spdlog::error("close(), errno: {}", errno);
        }
    }
    
    pid_t child_pid = fork(); //span a child process

    if (child_pid == -1) { // fork() failed, no child process created
        spdlog::error("fork() failed, errno: {}", errno);
        if (close(pipefd_err[0]) == -1 || close(pipefd_err[1]) == -1 ||
            close(pipefd_out[0]) == -1 || close(pipefd_out[1]) == -1) {
            spdlog::error("close(), errno: {}", errno);
        }
    }

    if (child_pid == 0) { // fork() succeeded, we are in the child process
        if (close(pipefd_out[0]) == -1) {
            spdlog::error("close(pipefd_stdout[0]) in child failed, errno: {}",
                errno);
        }
        if (close(pipefd_err[0]) == -1) {
            spdlog::error("close(pipefd_err[0]) in child failed, errno: {}",
                errno);
        }
        // man dup2
        if (dup2(pipefd_out[1], STDOUT_FILENO) == -1 ||
            dup2(pipefd_err[1], STDERR_FILENO) == -1) {
            spdlog::error("dup2() in child failed, some output may be missing, "
                "errno: {}", errno);
        }

        execv(cargs[0], cargs.data());
        spdlog::error("execv() in child failed, errno: {}", errno);
        // The exec() functions return only if an error has occurred.
        // The return value is -1, and errno is set to indicate the error.
        return;
    }
    
    //Only parent gets here
    if (close(pipefd_out[1]) == -1 || close(pipefd_err[1]) == -1) {
        spdlog::error("close() in parent failed, errno: {}", errno);
    }

    struct pollfd pfds[] = {
        { pipefd_out[0], POLLIN, 0 },
        { pipefd_err[0], POLLIN, 0 },
    };
    int nfds = sizeof(pfds) / sizeof(struct pollfd);    
    int open_fds = nfds;

    while (open_fds > 0) {
        if (poll(pfds, nfds, -1) == -1) {
            spdlog::error("poll() failed, some output may be missing, "
                "errno: {}", errno);
        }

        for (int j = 0; j < nfds; j++) {            
            if (pfds[j].revents != 0) {
                if (pfds[j].revents & POLLIN) {
                    ssize_t s = read(pfds[j].fd, buffer, sizeof(buffer)-1);
                    if (s == -1) {
                        spdlog::error("read() failed, "
                            "some output may be missing, errno: {}", errno);
                    }
                    if (j == 0) { stdoutStr += buffer; }
                    else { stderrStr += buffer; }
                } else {                /* POLLERR | POLLHUP */
                    open_fds--;
                }
            }
        }
    }

    int status;
    // wait for the child process to terminate
    if (waitpid(child_pid, &status, 0) == -1) {
        spdlog::error("waitpid() failed, errno: {}", errno);
        return;
    }
    if (WIFEXITED(status)) {
        rc = WEXITSTATUS(status);
    } else {
        rc = EXIT_FAILURE;
        if (WIFSIGNALED(status)) {
            spdlog::error("child process exited abnormally "
                "(terminated by a signal: {})", WTERMSIG(status));
        } else if (WIFSTOPPED(status)) {
            spdlog::error("child process exited abnormally "
                "(stopped by delivery of a signal: {})", WSTOPSIG(status));
        } else {
            spdlog::error("child process exited abnormally "
                "(unknown status: {})", status);
        }
    }
    return;
}

void execAsync(void* This, const vector<string>& args, exec_cb cb) {
    auto f = [](void* This, const vector<string>& args, exec_cb cb) {
        string stdout, stderr;
        int rc;
        exec(args, stdout, stderr, rc);
        cb(This, stdout, stderr, rc);
    };    
    thread th_exec(f, This, args, cb);
    th_exec.detach();
}
