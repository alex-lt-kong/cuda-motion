#include <sys/wait.h>
#include <iomanip>
#include <sstream>

#include <spdlog/spdlog.h>

#include "utils.h"

void exec(const vector<string>& args, string& stdout, string& stderr, int& rc) {

    vector<char*> cargs;
    cargs.reserve(args.size() + 1);

    for(size_t i = 0; i < args.size(); ++i) {
        cargs.push_back(const_cast<char*>(args[i].c_str()));
    }
    cargs.push_back(NULL);

    int pipefd_stdout[2], pipefd_stderr[2];
    // pipefd[0] is the read end of the pipe
    // pipefd[1] is the write end of the pipe
    FILE* stdoutFd;
    FILE* stderrFd;
    char buffer[4096];
    int status;

    if (pipe(pipefd_stdout) == -1) {      
        spdlog::error("pipe(pipefd_stdout) failed, errno: {}", errno);
        return;
    }
    if (pipe(pipefd_stderr) == -1) {
        close(pipefd_stdout[0]);
        close(pipefd_stdout[1]);
        spdlog::error("pipe(pipefd_stderr) failed, errno: {}", errno);
        return;
    }
    
    pid_t child_pid = fork(); //span a child process

    if (child_pid == -1) { // fork() failed, no child process created
        spdlog::error("fork() failed, errno: {}", errno);
        return;
    }

    if (child_pid == 0) { // fork() succeeded, we are in the child process
        close(pipefd_stdout[0]);
        close(pipefd_stderr[0]);
        // man dup2
        dup2(pipefd_stdout[1], STDOUT_FILENO);
        dup2(pipefd_stderr[1], STDERR_FILENO);

        execv(cargs[0], cargs.data());
        perror("execl()");
        // The exec() functions return only if an error has occurred.
        // The return value is -1, and errno is set to indicate the error.
        return;
    }
    
    //Only parent gets here
    close(pipefd_stdout[1]);
    close(pipefd_stderr[1]);

    if ((stdoutFd = fdopen(pipefd_stdout[0], "r")) == NULL) {
        spdlog::error("fdopen() failed, errno: {}", errno);
        return;
    }
    if ((stderrFd = fdopen(pipefd_stderr[0], "r")) == NULL) {
        fclose(stdoutFd);
        spdlog::error("fdopen() failed, errno: {}", errno);
        return;
    }

    while(fgets(buffer, sizeof(buffer) - 1, stdoutFd)) {            
        stdout += buffer;
    }
    while(fgets(buffer, sizeof(buffer) - 1, stderrFd)) {            
        stderr += buffer;
    }
    fclose(stdoutFd);
    fclose(stderrFd);

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
