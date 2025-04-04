#!/bin/bash
make clean
make
if [$1 == "-f"]; then
    echo "Running with fp trace"
    time ./cbp sample_traces/fp/sample_fp_trace.gz
else if [$1 == "-i"]; then
    echo "Running with int trace"
    time ./cbp sample_traces/int/sample_int_trace.gz
fi
