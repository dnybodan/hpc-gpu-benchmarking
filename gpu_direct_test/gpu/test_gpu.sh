#!/bin/bash

# Path to your CUDA executable
cuda_executable="./gpu_log"

# Run the CUDA program
$cuda_executable

# Assuming the CUDA program generates 'throughput.log'
# Now, call the Python script to plot
python ../scripts/plot_throughput.py

