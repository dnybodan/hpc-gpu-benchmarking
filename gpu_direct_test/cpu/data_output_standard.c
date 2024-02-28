#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>

#define BUFFER_SIZE (64 * 1024 * 1024) // 64 MB
#define NUM_ITERATIONS 100ll

long long current_time_microseconds() {
    struct timeval te; 
    gettimeofday(&te, NULL);
    return te.tv_sec * 1000000LL + te.tv_usec;
}

int main() {
    char *data = (char *)malloc(BUFFER_SIZE);
    int fd = open("/mnt/beegfs/standard_test/test_file", O_CREAT | O_WRONLY, 0664);

    if (fd < 0) {
        perror("Error opening file");
        free(data);
        return 1;
    }

    FILE *log_file = fopen("throughput_standard.log", "w");
    if (log_file == NULL) {
        perror("Cannot open log file");
        close(fd);
        free(data);
        return 1;
    }

    FILE *average_throughput_log_file = fopen("average_throughput_standard.log", "w");
    if (average_throughput_log_file == NULL) {
        perror("cannot open average log file");
        close(fd);
        free(data);
        return 1;
    }

    long long first_start = current_time_microseconds();
    double max_throughput = 0;
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        long long start_time = current_time_microseconds();

        ssize_t written = write(fd, data, BUFFER_SIZE);
        if (written < 0) {
            perror("Error writing to file");
            break;
        }

        long long end_time = current_time_microseconds();
        double time_taken = (end_time - start_time) / 1000000.0;

        if (time_taken > 0) {
            double throughput = BUFFER_SIZE / time_taken;
            fprintf(log_file, "%f\n", throughput);
            if (throughput > max_throughput) {
                max_throughput = throughput;
            }
        }
    }
    long long last_end = current_time_microseconds();

    double average_throughput = ((BUFFER_SIZE*NUM_ITERATIONS)*8/1000000000.0) / ((last_end-first_start)/1000000.0);

    printf("Maximum Throughput (Standard): %f Bytes/s\n", max_throughput);
    printf("Average Throughput over all %d iterations(Standard): %f Gbps", NUM_ITERATIONS, average_throughput);
    fprintf(average_throughput_log_file, "%f\n", average_throughput);
   
    close(fd);
    fclose(log_file);
    fclose(average_throughput_log_file);
    free(data);
    return 0;
}
