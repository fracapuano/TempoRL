#!/bin/bash
# activate conda environment for training
conda activate elienv

healthy_coeff="0.01 0.05 0.1 0.2 0.5"
performance_coeff="0.1 0.5 1 2"

for h in $healthy_coeff; do
    for p in $performance_coeff; do
        echo "training with healthy coeff: $h, performance coeff $p"
        python train.py --healthy-coeff $h --performance-coeff $p &
    done
done