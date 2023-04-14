<a href="https://ibb.co/qsKfhNN"><img src="https://i.ibb.co/hWbjrBB/tempo-RL-gale.jpg" alt="tempo-RL-gale" border="0"></a>

# [TempoRL: Laser Pulse Temporal Shape Optimization with Deep Reinforcement Learning](link_to_arxiv)
#### Francesco Capuano<sup>2</sup>, Davorin Peceli <sup>1</sup>, Gabriele Tiboni <sup>2</sup>, Raffaello Camoriano <sup>2</sup>, Bedrich Rus <sup>1</sup>
#### <sup>1</sup> ELI Beamlines, Dolní Břežany (Prague), Czech Republic, <sup>2</sup> VANDAL Laboratory, Politecnico di Torino, Turin, Italy

This repo contains the code for the paper [*TempoRL: laser pulse temporal shape optimization with Deep Reinforcement Learning*](link_to_arxiv), submitted at the SPIE Optics + Optoelectronics 2023 conference in Prague. 

*Abstract*: High Power Laser’s (HPL) optimal performance is essential for the success of a wide variety of experimental tasks related to light-matter interactions. Traditionally, HPL parameters are optimized in an automated fashion relying on black-box numerical methods. However, these can be demanding in terms of computational resources and usually disregard transient and complex dynamics. Model-free Deep Reinforcement Learning (DRL) offers a promising alternative framework for optimizing HPL performance since it allows to tune the control parameters as a function of system states subject to nonlinear temporal dynamics without requiring an explicit dynamics model of those. Furthermore, DRL aims to find an optimal control policy rather than a static parameter configuration, particularly suitable for dynamic processes involving sequential decision-making. This is particularly relevant as laser systems are typically characterized by dynamic rather than static traits. Hence the need for a strategy to choose the control applied based on the current context instead of one single optimal control configuration. This paper investigates the potential of DRL in improving the efficiency and safety of HPL control systems. We apply this technique to optimize the temporal profile of laser pulses in the L1 pump laser hosted at the ELI Beamlines facility. We show how to adapt DRL to the setting of spectral phase control by solely tuning dispersion coefficients of the spectral phase and reaching pulses similar to transform limited with full-width at half-maximum (FWHM) of ∼1.6 ps.

> An *extended abstract* can be found [here](link_to_extended_abstract).

This `README.md` files serves as a guide to reproduce our experiments and findings.

# Installation
Here the installation details are presented.
# How to train different agents
Here should go informations on how to launch the `train.py` script.
# How to test the different agents
Here should go informations on how to launch the `test.py` script with trained models in model.
# Cite us
If you use this repository, please consider citing

```bash
@misc{capuano23temporl,
  title = {TempoRL: laser pulse temporal shape optimization with Deep Reinforcement Learning},
  author = {Capuano, Francesco and Peceli, Davorin and Tiboni, Gabriele and Camoriano, Raffaello and Rus, Bedrich},
  doi = {HERE GOES DOI},
  publisher = {HERE GOES PUBLISHER},
  year = {2023}
}
```