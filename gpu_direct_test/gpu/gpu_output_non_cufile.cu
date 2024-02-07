#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define BUFFER_SIZE (64 * 1024 * 1024) // 64 MB

// Function to get current time in microseconds
long long current_time_microseconds() {
    struct timeval te; 
    gettimeofday(&te, NULL);
    return te.tv_sec * 1000000LL + te.tv_usec;
}

__global__ void generateDataKernel(char *data, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = (char)(idx % 256);
    }
}

int main() {
    char *d_data;  // Pointer for device (GPU) data
    char *h_data;  // Pointer for host (CPU) data

    // Allocate memory on GPU
    cudaMalloc((void **)&d_data, BUFFER_SIZE);

    // Allocate memory on host
    h_data = (char *)malloc(BUFFER_SIZE);

    // Initialize data on GPU
    generateDataKernel<<<(BUFFER_SIZE + 255) / 256, 256>>>(d_data, BUFFER_SIZE);
    cudaDeviceSynchronize();

    FILE *log_file = fopen("throughput_gpu_to_cpu.log", "w");
    if (log_file == NULL) {
        perror("Cannot open log file");
        return 1;
    }

    int fd = open("/mnt/beegfs/gpu_non_direct_test/testfile", O_CREAT | O_WRONLY, 0664);
    if (fd < 0) {
        perror("Error opening file");
        return 1;
    }

    for (int i = 0; i < 100; ++i) { 
        long long start_time = current_time_microseconds();

        // Copy data from GPU to host
        cudaMemcpy(h_data, d_data, BUFFER_SIZE, cudaMemcpyDeviceToHost);

        // Write data from host to file
        write(fd, h_data, BUFFER_SIZE);

        long long end_time = current_time_microseconds();
        double time_taken = (end_time - start_time) / 1000000.0; 

        if (time_taken > 0) {
            double throughput = BUFFER_SIZE / time_taken; 
            fprintf(log_file, "%f\n", throughput);
        }
    }

    fclose(log_file);
    close(fd);

    // Cleanup
    cudaFree(d_data);
    free(h_data);

    return 0;
}
