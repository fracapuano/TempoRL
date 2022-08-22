""" 
This script performs Bayesian Optimisation to optimize pulse shape.

Author: Francesco Capuano, ELI-beamlines intern, Summer 2022. 
"""
# these imports are necessary to import modules from directories one level back in the folder structure
import dill as pickle
import sys
import os
from turtle import up
from typing import Tuple
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, Bounds
from utils import physics
from utils import LaserModel as LM
from scipy.constants import c 

def extract_data()->Tuple[np.array, np.array]: 
    """This function extracts the desired information from the data file given.
    
    Returns: 
        Tuple[np.array, np.array]: Frequency (in THz) and Intensity arrays.

    """
    data_path = "../data/L1_pump_spectrum.csv"
    # read the data
    df = pd.read_csv(data_path, header = None)
    df.columns = ["Wavelength (nm)", "Intensity"]
    # converting Wavelength (nm) to Frequency (Hz)
    df["Frequency (THz)"] = df["Wavelength (nm)"].apply(lambda wavelenght: 1e-12 * (c/(wavelenght * 1e-9)))
    # clipping everything that is negative - measurement error
    df["Intensity"] = df["Intensity"].apply(lambda intensity: np.clip(intensity, a_min = 0, a_max = None))
    # the observations must be returned for increasing values of frequency
    df = df.sort_values(by = "Frequency (THz)")

    frequency, intensity = df.loc[:, "Frequency (THz)"].values, df.loc[:, "Intensity"].values
    # mapping intensity in the 0-1 range
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    
    return frequency, intensity

def model(frequency, intensity, compressor, b_int, cutoff)->object: 
    """
    Returns the model instantiated with respect to arguments. 
    TO BE REMOVED: Consistency of arguments type is assumed in this function.
    """
    l1_pump = LM.LaserModel(
        frequency = frequency, 
        intensity = intensity,
        compressor_params = compressor, 
        B = b_int, 
        cutoff = cutoff)

    l1_pump.preprocessing()
    
    return l1_pump
def bounds(GDDperc=0.05, TODperc=0.4, FODperc=0.3): 
    frequency, intensity = extract_data() # extracting the desired information
    COMPRESSOR = -1 * np.array((267.422 * 1e-24, -2.384 * 1e-36, 9.54893 * 1e-50)) # in s^2, s^3 and s^4 (SI units)
    B = 2
    CUTOFF = (289.95, 291.91) # cutoff frequencies, in THz

    l1_pump = model(frequency, intensity, COMPRESSOR, B, CUTOFF)
    # stretcher control bounds are centered in the compressor bounds and have a width related to a given percentage
    # (which can be though of as an hyperparameter as long as it is in the tunable interval)

    low_stretcher, high_stretcher = (-1 * COMPRESSOR * np.array((1 - GDDperc, 1 - TODperc, 1 - FODperc)), 
                                    -1 * COMPRESSOR * np.array((1 + GDDperc, 1 + TODperc, 1 + FODperc))
                                    )

    # stretcher control must be given in terms of dispersion coefficients so they must be translated into d2, d3 and d4. 
    low_stretcher, high_stretcher = (l1_pump.translate_control(low_stretcher, verse = "to_disp"),
                                    l1_pump.translate_control(high_stretcher, verse = "to_disp")
                                    )

    # ordering low_stretcher so that, row-wise, the first elements is always smaller than the second one
    bounds = np.sort(np.array([
        low_stretcher, 
        high_stretcher
    ]).T, axis = 1)

    low_stretcher, high_stretcher = bounds[:, 0], bounds[:, 1]
    # these are the bounds for the parameter currently optimized - sign can change so sorting is used
    pbounds = Bounds(
        lb = low_stretcher, 
        ub = high_stretcher
    )
    return pbounds

def objective_function(x,): 
    frequency, intensity = extract_data() # extracting the desired information
    COMPRESSOR = -1 * np.array((267.422 * 1e-24, -2.384 * 1e-36, 9.54893 * 1e-50)) # in s^2, s^3 and s^4 (SI units)
    B = 2
    CUTOFF = (289.95, 291.91) # cutoff frequencies, in THz

    l1_pump = model(frequency, intensity, COMPRESSOR, B, CUTOFF)
    # pre-processed version of frequency and intensity
    frequency_clean, intensity_clean = l1_pump.spit_center()

    _, profile_TL = physics.temporal_profile(frequency_clean, np.sqrt(intensity_clean), phase = np.zeros_like(frequency_clean), npoints_pad = l1_pump.pad_points)
    d2, d3, d4 = x
    temporal_profile = l1_pump.forward_pass(np.array((d2, d3, d4)))[1]
    temporal_profile = np.roll(temporal_profile, np.argmax(temporal_profile) - np.argmax(profile_TL))
    return physics.mse(temporal_profile, profile_TL)

def initial_guess(): 
    frequency, intensity = extract_data() # extracting the desired information
    COMPRESSOR = -1 * np.array((267.422 * 1e-24, -2.384 * 1e-36, 9.54893 * 1e-50)) # in s^2, s^3 and s^4 (SI units)
    B = 2
    CUTOFF = (289.95, 291.91) # cutoff frequencies, in THz

    l1_pump = model(frequency, intensity, COMPRESSOR, B, CUTOFF)
    x0 = l1_pump.translate_control(-1 * COMPRESSOR, verse = "to_disp")

    return x0

def main()->None: 

    b = bounds()
    init_guess = initial_guess()
    seed = 7

    # optimisation parameter
    recombination, mutation, tol, maxiter = 0.7, 0.6, 1e-5, 10
    
    result = differential_evolution(
        func = objective_function, 
        bounds = b, 
        maxiter = maxiter, 
        tol = tol, 
        seed = seed, 
        # x0 = init_guess, 
        workers = -1,
        recombination = recombination, 
        mutation = mutation, 
        disp = True
    )

    optimal_control = result.x
    optimald2, optimald3, optimald4 = optimal_control

    frequency, intensity = extract_data() # extracting the desired information
    COMPRESSOR = -1 * np.array((267.422 * 1e-24, -2.384 * 1e-36, 9.54893 * 1e-50)) # in s^2, s^3 and s^4 (SI units)
    B = 2
    CUTOFF = (289.95, 291.91) # cutoff frequencies, in THz

    l1_pump = model(frequency, intensity, COMPRESSOR, B, CUTOFF)
    # pre-processed version of frequency and intensity
    frequency_clean, intensity_clean = l1_pump.spit_center()

    time, profile_TL = physics.temporal_profile(frequency_clean, np.sqrt(intensity_clean), phase = np.zeros_like(frequency_clean), npoints_pad = l1_pump.pad_points)

    # applying optimal control found
    time_BO, profile_BO = l1_pump.forward_pass(optimal_control)

    fig, ax = plt.subplots()
    
    ax.set_title("Pulse Optimization results", fontsize = 12, fontweight = "bold")
    ax.plot(time_BO, profile_BO, label = "DE output")
    ax.scatter(time, profile_TL, label = "Target Pulse", c = "tab:grey", marker = "x", s = 50)
    ax.set_xlabel(r"Time (s)"); ax.set_ylabel("Intensity")

    ax.legend()
    ax.set_xlim(left = -0.1e-11, right = +0.1e-11)
    fig.tight_layout()
    plt.show()

    print("Final MSE between Control and Target: {:.4e}".format(-1*objective_function(optimald2, optimald3, optimald4)))
 
if __name__ == "__main__": 
    main()