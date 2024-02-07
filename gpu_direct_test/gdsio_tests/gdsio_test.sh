#!/bin/bash

log_file="gdsio_output.log"

# Empty the log file if it already exists
> $log_file

for i in {1..50}; do
    # Run the command and append the output to the log file
    /usr/local/cuda/gds/tools/gdsio -D /mnt/beegfs/gdsio -w 32 -d 1 -I 1 -x 5 -s 1G -i 1Md >> $log_file
    echo "" >> $log_file # Adding an empty line for separation between runs
done

# Call the Python script to process and plot the data
python3 process_and_plot.py $log_file
