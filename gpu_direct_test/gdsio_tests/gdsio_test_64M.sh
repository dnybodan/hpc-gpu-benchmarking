
g_file="../throughput_gdsio.log"
test_file="/mnt/beegfs/gdsio/test_file"
buffer_size=67108864 # 64MB in bytes
iterations=50

# Empty the log file if it already exists
> "$log_file"

for i in $(seq 1 $iterations); do
    start_time=$(date +%s%N) # current time in nanoseconds

    # Run the GDSIO command. Adjust parameters as needed for your test.
    /usr/local/cuda/gds/tools/gdsio -D "$test_file" -w 32 -d 1 -I 1 -x 5 -s 64M -i 1 >> /dev/null

    end_time=$(date +%s%N) # end time in nanoseconds
    time_taken=$((end_time - start_time))
    time_taken_seconds=$(echo "scale=9; $time_taken/1000000000" | bc) # Convert nanoseconds to seconds

    if (( $(echo "$time_taken_seconds > 0" | bc -l) )); then
        throughput=$(echo "scale=2; $buffer_size / $time_taken_seconds" | bc)
        echo "$throughput" >> "$log_file"
    fi
done

echo "GDSIO test completed. Throughput results saved to $log_file."

