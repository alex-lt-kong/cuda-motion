#include <sys/wait.h>
#include <iomanip>
#include <sstream>

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
    char buffer[256];
    int status;

    if (pipe(pipefd_stdout) == -1) {
        throw runtime_error("pipe(pipefd_stdout) failed, errno: " + to_string(errno));
    }
    if (pipe(pipefd_stderr) == -1) {
        close(pipefd_stdout[0]);
        close(pipefd_stdout[1]);
        throw runtime_error("pipe(pipefd_stderr) failed, errno: " + to_string(errno));
    }
    
    pid_t child_pid = fork(); //span a child process

    if (child_pid == -1) { // fork() failed, no child process created
        throw runtime_error("fork() failed, errno: " + to_string(errno));
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
        throw runtime_error("fdopen() failed, errno: " + to_string(errno));
    }
    if ((stderrFd = fdopen(pipefd_stderr[0], "r")) == NULL) {
        fclose(stdoutFd);
        throw runtime_error("fdopen() failed, errno: " + to_string(errno));
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
    waitpid(child_pid, &status, 0);
    if (WIFEXITED(status)) {
        rc = WEXITSTATUS(status);
    } else {
        rc = -1;
    }
}

void exec_async(void* This, const vector<string>& args, exec_cb cb) {
    auto f = [](void* This, const vector<string>& args, exec_cb cb) {
        string stdout, stderr;
        int rc;
        exec(args, stdout, stderr, rc);
        cb(This, stdout, stderr, rc);
    };    
    thread th_exec(f, This, args, cb);
    th_exec.detach();
}
