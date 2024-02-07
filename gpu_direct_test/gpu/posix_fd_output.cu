#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <sys/time.h>
#include <errno.h>

#define BUFFER_SIZE (64 * 1024 * 1024) // 4 KB

// Function to get current time in microseconds
long long current_time_microseconds() {
    struct timeval te;
    gettimeofday(&te, NULL);
    long long microseconds = te.tv_sec * 1000000LL + te.tv_usec;
    return microseconds;
}

int main() {
    int fd;
    const char *filePath = "/mnt/beegfs/posix_test/test_file";

    // Open the file with POSIX O_DIRECT flag to bypass the page cache
    fd = open(filePath, O_CREAT | O_RDWR | O_DIRECT, 0664);
    if (fd < 0) {
        fprintf(stderr, "Error opening file %s errno %d\n", filePath, errno);
        return -1;
    }

    // Allocate aligned memory to satisfy O_DIRECT requirements
    char *buf;
    posix_memalign((void **)&buf, BUFFER_SIZE, BUFFER_SIZE); // Align to the buffer size

    // Simulate generating data to write to the file (replacing the CUDA kernel)
    for (int i = 0; i < BUFFER_SIZE; ++i) {
        buf[i] = (char)(i % 256);
    }

    FILE *log_file = fopen("throughput_posix.log", "w");
    if (log_file == NULL) {
        perror("Cannot open log file");
        close(fd);
        free(buf);
        return 1;
    }

    // Write data to file and log throughput
    for (int i = 0; i < 100; ++i) {
        long long start_time = current_time_microseconds();

        ssize_t written = write(fd, buf, BUFFER_SIZE);
        if (written < 0) {
            fprintf(stderr, "Error writing to file: %s\n", strerror(errno));
            break;
        }

        long long end_time = current_time_microseconds();
        double time_taken = (end_time - start_time) / 1000000.0; // Convert to seconds

        if (time_taken > 0) {
            double throughput = BUFFER_SIZE / time_taken; // bytes per second
            fprintf(log_file, "%f\n", throughput);
        }
    }

    printf("Done writing. Cleaning up.\n");

    fclose(log_file);
    close(fd);
    free(buf);

    return 0;
}

