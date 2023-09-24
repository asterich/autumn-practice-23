const char *perl_program = "use 5.020;\nuse utf8;\nuse warnings;\nuse autodie;\nuse feature 'signatures';\nuse List::Util qw/min/;\n\nmy $vertex_num = $ARGV[0];\n\nmy @result = ();\nfor (0 .. $vertex_num - 1) {\n    push @result, [];\n}\n\nfor my $i (0 .. $vertex_num - 1) {\n    for my $j (0 .. $vertex_num - 1) {\n        my $tmp;\n        sysread STDIN, $tmp, 4;\n        $result[$i][$j] = unpack 'l', $tmp;\n    }\n}\n\nfor my $k (0 .. $vertex_num - 1) {\n    for my $i (0 .. $vertex_num - 1) {\n        for my $j (0 .. $vertex_num - 1) {\n            $result[$i][$j] = min $result[$i][$j], $result[$i][$k] + $result[$k][$j];\n        }\n    }\n}\n\nfor my $i (0 .. $vertex_num - 1) {\n    for my $j (0 .. $vertex_num - 1) {\n        my $buf = pack 'l', $result[$i][$j];\n        syswrite STDOUT, $buf, 4;\n    }\n}\n\n\nsay STDERR $result[0][0];\n\n__DATA__\nGraph Graph::apsp() {\n    Graph result(*this);\n    for (int k = 0; k < vertex_num_; ++k) {\n        for (int i = 0; i < vertex_num_; ++i) {\n            for (int j = 0; j < vertex_num_; ++j) {\n                result(i, j) = std::min(result(i, j), result(i, k) + result(k, j));\n            }\n        }\n    }\n    return result;\n}\n";
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
