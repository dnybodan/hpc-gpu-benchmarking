import matplotlib.pyplot as plt

def read_throughput_data(file_path):
    with open(file_path, 'r') as file:
        return [float(line.strip()) for line in file]

def plot_throughput(data):
    plt.plot(data)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Throughput (Gbps)')
    plt.title('Write Throughput Over Time')
    plt.show()

def main():
    data = read_throughput_data('../gpu/throughput.log')
    plot_throughput(data*8)

if __name__ == "__main__":
    main()

