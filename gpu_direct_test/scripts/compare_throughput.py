import matplotlib.pyplot as plt

def read_throughput_data(file_path):
    with open(file_path, 'r') as file:
        return [float(line.strip()) for line in file]

def plot_comparison(standard_data, cufile_data,posix_data,  gpu_to_cpu_data):
    plt.plot([8*x/1000000000 for x in standard_data], label='Standard I/O')
    plt.plot([8*x/1000000000 for x in gpu_to_cpu_data], label='GPU to CPU I/O') 
    plt.plot([8*x/1000000000 for x in posix_data], label='POSIX MEMSET O_DIRECT I/O')
    plt.plot([8*x/1000000000 for x in cufile_data], label='cuFile I/O')
    plt.xlabel('Iteration')
    plt.ylabel('Throughput (Gbps)')
    plt.title('Throughput Comparison')
    plt.legend()
    plt.savefig('throughput_comparison_open_method.png')
    plt.show()

def main():
    standard_data = read_throughput_data('throughput_standard.log')
    cufile_data = read_throughput_data('throughput_cufile.log')
    gpu_to_cpu_data = read_throughput_data('throughput_gpu_to_cpu.log')
    posix_data = read_throughput_data('throughput_posix.log')
    plot_comparison(standard_data, cufile_data, posix_data,gpu_to_cpu_data)

if __name__ == "__main__":
    main()
