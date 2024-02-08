import sys
import matplotlib.pyplot as plt
import re

def process_and_plot(log_file):
    with open(log_file, 'r') as file:
        lines = file.readlines()

    throughputs = []
    latencies = []

    for line in lines:
        # Extracting throughput and latency using regular expressions
        if "Throughput:" in line:
            throughput_match = re.search(r"Throughput:\s*([\d.]+)\s*GiB/sec,", line)
            if throughput_match:
                throughputs.append(float(throughput_match.group(1)))
        
        if "Avg_Latency:" in line:
            latency_match = re.search(r"Avg_Latency:\s*([\d.]+)", line)
            if latency_match:
                latencies.append(float(latency_match.group(1)))

    # Plotting Throughput
    plt.figure(1)
    plt.plot([8.59*x for x in throughputs])
    plt.title('Throughput over Runs')
    plt.xlabel('Run (33 GB Write)')
    plt.ylabel('Throughput (Gbps)')
    plt.savefig('throughput_gdsio_plot.png')

    # Plotting Latency
    plt.figure(2)
    plt.plot(latencies)
    plt.title('Latency over Runs')
    plt.xlabel('Run (33 GB Write)')
    plt.ylabel('Latency (us)')
    plt.savefig('latency_plot.png')

    plt.show()

if __name__ == "__main__":
    log_file = sys.argv[1]
    process_and_plot(log_file)
