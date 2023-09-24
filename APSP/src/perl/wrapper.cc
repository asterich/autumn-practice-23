#include "graph.hh"
#include <unistd.h>
#include <string.h>
#include <sys/wait.h>

Graph Graph::apsp() {
    Graph result(*this);

    char vertex_num_param[512];
    sprintf(vertex_num_param, "%d", vertex_num_);

    int send_fd[2];
    pipe(send_fd);

    int recv_fd[2];
    pipe(recv_fd);

    int pid = fork();
    if (pid == 0) {
        close(send_fd[0]);
        close(recv_fd[1]);
    } else {
        close(send_fd[1]);
        close(recv_fd[0]);
        dup2(send_fd[0], STDIN_FILENO);
        dup2(recv_fd[1], STDOUT_FILENO);
        execlp("perl", "perl", "-e", perl_program, vertex_num_param, NULL);
    }

    for (int i = 0; i < vertex_num_; ++i) {
        for (int j = 0; j < vertex_num_; ++j) {
            int tmp = result(i, j);
            write(send_fd[1], &tmp, sizeof(tmp));
        }
    }

    for (int i = 0; i < vertex_num_; ++i) {
        for (int j = 0; j < vertex_num_; ++j) {
            int tmp;
            read(recv_fd[0], &tmp, sizeof(tmp));
            result(i, j) = tmp;
        }
    }

    waitpid(pid, NULL, 0);

    return result;
}
