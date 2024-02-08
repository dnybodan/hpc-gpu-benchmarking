CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS= -I/usr/local/cuda/include -L/usr/local/cuda-12.1/lib64 -lcufile
PYTHON=python

# Replace with your actual CUDA file and standard C file names
CUDA_SOURCE=gpu/gpu_output_cufile.cu
C_SOURCE=cpu/data_output_standard.c
PYTHON_SCRIPT=scripts/compare_throughput.py
GPU_TO_CPU_SOURCE=gpu/gpu_output_non_cufile.cu
POSIX =gpu/posix_fd_output.cu

all: run_standard run_cuda run_gpu_to_cpu run_posix run_gdsio plot

data_output_standard: $(C_SOURCE)
	$(CC) -o $@ $^ $(CFLAGS)

data_output_cufile: $(CUDA_SOURCE)
	$(NVCC) -o $@ $^ $(NVCCFLAGS)

data_output_non_cufile: $(GPU_TO_CPU_SOURCE)
	$(NVCC) -o $@ $^ $(NVCCFLAGS)

data_output_posix: $(POSIX)
	$(NVCC) -o $@ $^ $(NVCCFLAGS)

run_standard: data_output_standard
	./data_output_standard

run_cuda: data_output_cufile
	./data_output_cufile

run_gpu_to_cpu: data_output_non_cufile
	./data_output_non_cufile

run_posix: data_output_posix
	./data_output_posix

run_gdsio:
	$(shell gdsio_tests/gdsio_test_64M.sh)

plot: $(PYTHON_SCRIPT)
	$(PYTHON) $(PYTHON_SCRIPT)

clean:
	rm -f data_output_standard data_output_cufile data_output_non_cufile data_output_posix throughput_posix.log throughput_standard.log throughput_gdsio.log throughput_cufile.log throughput_gpu_to_cpu.log throughput_comparison.png

.PHONY: all run_standard run_cuda run_gpu_to_cpu run_posix plot clean
