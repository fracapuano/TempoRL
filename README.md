<a href="https://ibb.co/qsKfhNN"><img src="https://i.ibb.co/hWbjrBB/tempo-RL-gale.jpg" alt="tempo-RL-gale" border="0"></a>

# [TempoRL: Laser Pulse Temporal Shape Optimization with Deep Reinforcement Learning](https://arxiv.org/abs/2304.12187)
#### Francesco Capuano<sup>2</sup>, Davorin Peceli <sup>1</sup>, Gabriele Tiboni <sup>2</sup>, Raffaello Camoriano <sup>2</sup>, Bedrich Rus <sup>1</sup>
#### <sup>1</sup> ELI Beamlines, Dolní Břežany (Prague), Czech Republic, <sup>2</sup> VANDAL Laboratory, Politecnico di Torino, Turin, Italy

This repo contains the code for the paper [*TempoRL: laser pulse temporal shape optimization with Deep Reinforcement Learning*](https://arxiv.org/abs/2304.12187), submitted at the SPIE Optics + Optoelectronics 2023 conference in Prague. 

*Abstract*: High Power Laser’s (HPL) optimal performance is essential for the success of a wide variety of experimental tasks related to light-matter interactions. Traditionally, HPL parameters are optimized in an automated fashion relying on black-box numerical methods. However, these can be demanding in terms of computational resources and usually disregard transient and complex dynamics. Model-free Deep Reinforcement Learning (DRL) offers a promising alternative framework for optimizing HPL performance since it allows to tune the control parameters as a function of system states subject to nonlinear temporal dynamics without requiring an explicit dynamics model of those. Furthermore, DRL aims to find an optimal control policy rather than a static parameter configuration, particularly suitable for dynamic processes involving sequential decision-making. This is particularly relevant as laser systems are typically characterized by dynamic rather than static traits. Hence the need for a strategy to choose the control applied based on the current context instead of one single optimal control configuration. This paper investigates the potential of DRL in improving the efficiency and safety of HPL control systems. We apply this technique to optimize the temporal profile of laser pulses in the L1 pump laser hosted at the ELI Beamlines facility. We show how to adapt DRL to the setting of spectral phase control by solely tuning dispersion coefficients of the spectral phase and reaching pulses similar to transform limited with full-width at half-maximum (FWHM) of ∼1.6 ps.

> An **extended abstract** and **live demo** can be found [here](https://sites.google.com/view/temporl-opt).

This `README.md` files serves as a guide to reproduce our experiments and findings.

# Installation
To reproduce our findings you will need to setup a virtual environment with the required dependancies (contained in `requirements.txt`). 
If you have `conda` installed in your system you can do this by running:
```bash
conda create --name <env_name> --file requirements.txt
```
# How to train different agents
We built this project with the goal of having something as modular as possible, so there are lots of configurations for the `train.py` file! These are mainly handled through arguments to the training file. You can have a look at these by simply running: 
```bash
python train.py --help
```
You can use the default configuration by simply running: 
```bash
python train.py --default
```
When doing this, you will trigger the training of a **PPO**-based agent on `MDP-v1` for **200k timesteps**, evaluating the policy **through wandb** every **10k timesteps** for **25 test-episodes**.

We considered implementing a simple bash script for the reader convenience to exactly reproduce our training pipeline for our best performing MDP (`MDP-2`). You can do this by running: 
```bash
bash do_training.sh
```
> Note: This command will trigger the execution of ~90 independent processes on the same amount of CPUs.

# How to test the different agents
We also provide already trained models in the `models/` folder. As we reported in the paper, our best performing model is **SAC**. You can reproduce the testing procedure by running: 
```bash
python test.py --default
```
Of course, you can choose whether to render or not the agent. You can specify this (and even plug your own custom agent) via the `test.py` args. To have a comprehensive view of those, you can run: 
```bash
python test.py --help
```

# Cite us
If you use this repository, please do cite:

```bash
@misc{capuano23temporl,
  title = {TempoRL: laser pulse temporal shape optimization with Deep Reinforcement Learning},
  author = {Capuano, Francesco and Peceli, Davorin and Tiboni, Gabriele and Camoriano, Raffaello and Rus, Bedrich},
  doi = {10.48550/arXiv.2304.12187},
  publisher = {arXiv},
  year = {2023}
}
```