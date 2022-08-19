""" 
This script performs Bayesian Optimisation to optimize pulse shape.

Author: Francesco Capuano, ELI-beamlines intern, Summer 2022. 
"""
# these imports are necessary to import modules from directories one level back in the folder structure
import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from utils import physics
from scipy.interpolate import interp1d

def main()->None: 
    filename = "LLNL_160809_freq.csv"
    os.chdir("..") # one folder back in the folder structure
    df = pd.read_csv("data/"+filename, header = None)
    df.columns = ["Frequency (in THz)", "Wavelength (in nm)", "Intensity", "Phase (rad)", "Phase (cutted) (rad)"] 

    frequency = df.loc[:, "Frequency (in THz)"].values * 10**12 # THz to Hz
    field = np.sqrt(df.loc[:, "Intensity"].values)

    # this must be be computed in advance for each given spectral intensity signal. 
    central_carrier = 2294295618320813.0

    # hyper parameters of the spectrum considered - These two must be computed for each spectral intensity signal. 
    num_points = int(1e3)
    start_freq, end_freq = 351, 379


    field_interpolator = interp1d(frequency, field)
    frequency_spaced = np.linspace(start = start_freq, stop = end_freq, num = num_points, endpoint = True) * 10**12
    field_spaced = field_interpolator(frequency_spaced)

    frequency = frequency_spaced; field = field_spaced

    # the control in this script is only done considering GDD, TOD and FOD.
    ps = physics.PulseEmitter(frequency = frequency, field = field, useEquation=True, central_carrier=central_carrier)

    # these are the bounds for the parameter currently optimized. These are specific for the actual laser rather than being specific for the signal.
    GDD_low, GDD_high = 20, 40
    TOD_low, TOD_high = 10, 100
    FOD_low, FOD_high = 10, 100

    pbounds = {
        "GDD": (GDD_low, GDD_high), 
        "TOD": (TOD_low, TOD_high), 
        "FOD": (FOD_low, FOD_high)
    }
    n_params = 3 # GDD, TOD and FOD
    # target pulse is obtained using zero phase - obtained using all zero coefficients in phase reconstruction
    
    theorical_phase = physics.theorical_phase(frequency, central_carrier)
    control_params = physics.phase_expansions(frequency, theorical_phase)

    psPhaseEquation = physics.PulseEmitter(frequency, field)
    target_pulse = psPhaseEquation.temporal_profile(np.zeros_like(control_params))
    timescale = psPhaseEquation.time_scale()

    objective_function = lambda GDD, TOD, FOD: -1 * physics.mse(ps.temporal_profile(control_params = np.array([GDD, TOD, FOD])), target_pulse) # the problem is a minimization one. 

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=7,
    )
    
    # these are hyperparameters of the optimization process
    n_init, n_iter = 100, 250
    optimizer.maximize(
        init_points=n_init,
        n_iter=n_iter,
    )

    # plotting the result obtained with Bayesian Optimization
    optimalGDD, optimalTOD, optimalFOD = optimizer.max["params"]["GDD"], optimizer.max["params"]["TOD"], optimizer.max["params"]["FOD"]
    optimal_control = np.array([optimalGDD, optimalTOD, optimalFOD])

    fig, ax = plt.subplots()
    ax.set_title("Pulse Optimization results")
    ax.plot(timescale, ps.temporal_profile(optimal_control), label = "Bayesian Optimization output")
    ax.scatter(timescale, target_pulse, label = "Target Pulse", c = "tab:grey", marker = "x", s = 50)
    ax.set_xlabel(r"Time ($10^{-12}$ s)"); ax.set_ylabel("Intensity")

    ax.legend()
    ax.set_xlim(left = -400, right = +400)
    fig.tight_layout()

    print("Final MSE between Control and Target: {:.4e}".format(-1*objective_function(optimalGDD, optimalTOD, optimalFOD)))

    plt.show()
if __name__ == "__main__": 
    main()