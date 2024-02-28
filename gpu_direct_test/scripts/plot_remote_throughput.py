import matplotlib.pyplot as plt
import os
import glob

def read_throughput_data(directory):
    data = {}
    file_pattern = os.path.join(directory, "*_throughput_standard.log")
    for file_path in glob.glob(file_pattern):
        hostname = os.path.basename(file_path).split('_')[0]
        with open(file_path, 'r') as file:
            data[hostname] = [float(line.strip()) for line in file]
    return data

def plot_comparison(data):
    plt.figure(figsize=(10, 6))
    for hostname, throughput in data.items():
        plt.plot([8*x/1000000000 for x in throughput], label=f'{hostname} Standard I/O')
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Throughput (Gbps)')
    plt.title('Throughput Comparison')
    plt.legend()
    plt.savefig('../plots/1_25_remote_hosts_throughput_comparison_standard.png')
    # plt.show()

def main():
    log_directory = "../remote_logs/"
    data = read_throughput_data(log_directory)
    plot_comparison(data)

if __name__ == "__main__":
    main()

