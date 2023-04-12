<a href="https://ibb.co/kXrcSY1"><img src="https://i.ibb.co/3c8hCtz/tempo-RL-gale.jpg" alt="tempo-RL-gale" border="0"></a>

# [TempoRL Optimization: Laser Temporal Shape Tuning with Deep Reinforcement Learning](link_to_arxiv)
#### Francesco Capuano<sup>2</sup>, Davorin Peceli <sup>1</sup>, Gabriele Tiboni <sup>2</sup>, Raffaello Camoriano <sup>2</sup>, Bedrich Rus <sup>1</sup>
#### <sup>1</sup> ELI Beamlines, Dolní Břežany (Prague), Czech Republic, <sup>2</sup> VANDAL Laboratory, Politecnico di Torino, Turin, Italy

This repo contains the code for the paper [*TempoRL optimization: laser temporal shape tuning with deep reinforcement learning*](link_to_arxiv), submitted at the SPIE Optics + Optoelectronics 2023 conference in Prague. 

*Abstract*: High Power Laser’s (HPL) optimal performance is essential for the success of a wide variety of experimental tasks. Traditional approaches to HPL parameters optimization rely on black-box numerical methods. However, these can be demanding in terms of computational resources, and usually disregard transient dynamics in presence of high-frequency pulse sequences. Model-free Deep Reinforcement Learning (DRL) offers a promising alternative framework for optimizing HPL performance since it allows to optimize control parameters as a function of system states subject to nonlinear temporal dynamics without requiring an explicit dynamics model. Furthermore, DRL aims at developing an optimal behavior rather than obtaining a static parameter configuration. This paper investigates the potential of DRL for improving the efficiency and safety of HPL systems control. We apply this technique to shape the temporal profile of laser pulses in the L1 pump laser hosted at the ELI ERIC facility. We show how to adapt DRL to the setting of spectral phase control by solely tuning dispersion coefficients of the spectral phase and reaching pulses similar to transform limited with full-width at half-maximum (FWHM) of ∼1.6 ps.

> 🚀 You can find an extended abstract [here](link_to_extended_abstract).

This `README.md` files serves as a guide to reproduce our experiments and findings.

