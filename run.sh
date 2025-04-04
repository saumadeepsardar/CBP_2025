#!/bin/bash
make clean
make

if [ "$1" == "-f" ]; then
    echo "Running with fp trace"
    time ./cbp sample_traces/fp/sample_fp_trace.gz
elif [ "$1" == "-i" ]; then
    echo "Running with int trace"
    time ./cbp sample_traces/int/sample_int_trace.gz
else
    echo "Usage: $0 [-f | -i]"
fi
