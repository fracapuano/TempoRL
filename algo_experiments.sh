#!/bin/bash

healthy_coeff="0"
# seeds used, just for readability
seeds="42 43 44 45 46"
# discount factors
gammas="0.7 0.8 0.9"

# this will launch (5 x 3) independent training processes. Each training process will require access to ~6 cores for the vectorized envs.
for gamma in $gammas; do
    python train.py --verbose 0 --train-timesteps 7e5 --env-version v2 --algorithm $1 --n-envs 6 --gamma $gamma --seed 42 --healthy-coeff $healthy_coeff&
    python train.py --verbose 0 --train-timesteps 7e5 --env-version v2 --algorithm $1 --n-envs 6 --gamma $gamma --seed 43 --healthy-coeff $healthy_coeff&
    python train.py --verbose 0 --train-timesteps 7e5 --env-version v2 --algorithm $1 --n-envs 6 --gamma $gamma --seed 44 --healthy-coeff $healthy_coeff&
    python train.py --verbose 0 --train-timesteps 7e5 --env-version v2 --algorithm $1 --n-envs 6 --gamma $gamma --seed 45 --healthy-coeff $healthy_coeff&
    python train.py --verbose 0 --train-timesteps 7e5 --env-version v2 --algorithm $1 --n-envs 6 --gamma $gamma --seed 46 --healthy-coeff $healthy_coeff&
done