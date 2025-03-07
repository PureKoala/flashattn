#!/bin/bash

loop_count=100
# thread_count=10

output_dir="attention-results"
mkdir -p $output_dir

for i in $(seq 1 $loop_count); do
    N=$((i * 64))
    echo "Running N = $N"
    ./build/AttnPThread $i >> $output_dir/attention-N-$N.txt &
done

wait
echo "All iterations completed."