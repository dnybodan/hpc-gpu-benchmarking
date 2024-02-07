#!/bin/bash

# Directory for FIO test
DIR="/mnt/beegfs/fiotest/"

# Create directory if it doesn't exist
if [ ! -d "$DIR" ]; then
    echo "Directory $DIR does not exist. Creating it..."
    mkdir -p "$DIR"
    if [ $? -ne 0 ]; then
        echo "Failed to create directory $DIR. Please check permissions."
        exit 1
    fi
fi

# FIO parameters
SIZE="50G" # Size of the test file
FILENAME="${DIR}fiotestfile" # Test file name

# FIO command for write test
fio --name=write_test \
    --filename=$FILENAME \
    --size=$SIZE \
    --ioengine=libaio \
    --rw=write \
    --bs=1M \
    --numjobs=25 \
    --direct=1 \
    --sync=0 \
    --randrepeat=0 \
    --iodepth=128 \
    --runtime=60 \
    --time_based \
    --group_reporting
