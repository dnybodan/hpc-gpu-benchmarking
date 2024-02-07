#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "cufile.h"

#define BUFFER_SIZE (64 * 1024 * 1024) // 4 KB

// Function to get current time in microseconds
long long current_time_microseconds() {
    struct timeval te; 
    gettimeofday(&te, NULL);
    long long microseconds = te.tv_sec * 1000000LL + te.tv_usec;
    return microseconds;
}

__global__ void generateDataKernel(char *data, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = (char)(idx % 256);
    }
}

int main() {
    CUfileError_t status;
    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;
    char *d_data;

    // Initialize GPU
    cudaMalloc((void **)&d_data, BUFFER_SIZE);
    generateDataKernel<<<(BUFFER_SIZE + 255) / 256, 256>>>(d_data, BUFFER_SIZE);

    // Initialize libcufile
    cuFileDriverOpen();
    memset((void *)&cf_descr, 0, sizeof(CUfileDescr_t));
    cf_descr.handle.fd = open("/mnt/beegfs/gpu_direct_test/test_file", O_CREAT | O_RDWR | O_DIRECT, 0664);
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    status = cuFileHandleRegister(&cf_handle, &cf_descr);

    if (status.err != CU_FILE_SUCCESS) {
        printf("Error: Failed to open file with cuFile\n");
        return -1;
    }

    FILE *log_file = fopen("throughput.log", "w");
    if (log_file == NULL) {
        perror("Cannot open log file");
        return 1;
    }

    // Write data to file using libcufile and log throughput
    for (int i = 0; i < 10; ++i) { // Example loop - adjust as needed
        long long start_time = current_time_microseconds();

        ssize_t ret = cuFileWrite(cf_handle, d_data, BUFFER_SIZE, 0, 0);
        if (ret < 0) {
            printf("Error: Failed to write to file\n");
            break;
        }

        long long end_time = current_time_microseconds();
        double time_taken = (end_time - start_time) / 1000000.0; // Convert to seconds

        if (time_taken > 0) {
            double throughput = BUFFER_SIZE / time_taken; // bytes per second
            fprintf(log_file, "%f\n", throughput);
        }
    }

    fclose(log_file);

    // Cleanup
    cuFileHandleDeregister(cf_handle);
    cudaFree(d_data);
    cuFileDriverClose();

    return 0;
}

