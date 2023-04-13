#!/bin/bash

algorithms="PPO TRPO SAC"

for a in $algorithms; do
    bash algo_experiments.sh $algo
done