""" 
This script performs Bayesian Optimisation to optimize pulse shape.

Author: Francesco Capuano, ELI-beamlines intern, Summer 2022. 
"""
# these imports are necessary to import modules from directories one level back in the folder structure
import sys
import os
from turtle import up
from typing import Mapping, Tuple
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, Bounds
from scipy.interpolate import UnivariateSpline
from utils import physics
from utils.se import get_project_root
from utils import LaserModel as LM
from scipy.constants import c
from scipy.signal import find_peaks, peak_widths

def extract_data()->Tuple[np.array, np.array]: 
    """This function extracts the desired information from the data file given.
    
    Returns: 
        Tuple[np.array, np.array]: Frequency (in THz) and Intensity arrays.

    """
    data_path = str(get_project_root()) + "/data/L1_pump_spectrum.csv"
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

def model(frequency, intensity, compressor, b_int, cutoff, num_points)->object: 
    """
    Returns the model instantiated with respect to arguments. 
    TO BE REMOVED: Consistency of arguments type is assumed in this function.
    """
    l1_pump = LM.LaserModel(
        frequency = frequency, 
        intensity = intensity,
        compressor_params = compressor, 
        B = b_int, 
        cutoff = cutoff,
        num_points = num_points)

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

def initial_guess() -> np.array: 
    """This function returns the translated version of the control parameters of the compressor so as to use them 
    as initial guess for differential evolution.

    Returns:
        np.array: Initial guess
    """
    frequency, intensity = extract_data() # extracting the desired information
    COMPRESSOR = -1 * np.array((267.422 * 1e-24, -2.384 * 1e-36, 9.54893 * 1e-50)) # in s^2, s^3 and s^4 (SI units)
    B = 2
    CUTOFF = (289.95, 291.91) # cutoff frequencies, in THz

    l1_pump = model(frequency, intensity, COMPRESSOR, B, CUTOFF)
    x0 = l1_pump.translate_control(-1 * COMPRESSOR, verse = "to_disp")

    return x0

def diff_evolution(
    objective_function:Mapping[np.array, float], 
    bounds:np.ndarray, 
    mutation:float = 0.8, 
    cross_p:float = 0.7, 
    population_size:int = 20, 
    maxit:int=int(3e3), 
    verbose:int=0, 
    print_every:int=10):
    """This function implements the Differential Evolution algorithm rand/1/bin version.

    Args:
        objective_function (function): Objective function to minimize.
        bounds (np.ndarray): Multi-dimensional array of dimension nx2, where n is the dimension of the input vector to objective function.
        mutation (float, optional): Mutation coefficient used to combine population elements. Must be in [0.5, 2). Defaults to 0.8.
        cross_p (float, optional): Probability that each element in the best candidate is replaced with the corresponding element in the mutant. 
                                   Defaults to 0.7.
        popsize (int, optional): Number of elements in the population. Defaults to 20.
        maxit (int, optional): Maximal number of iteration. Defaults to int(3e3).
        verbose (int, optional): Whether or not to display information along training. If larger than zero prints iteration
                                 number and objective function value. Defaults to 0.
        print_every (int, optional): Batch counter after which to print information if verbose is larger than 0. Defaults to 10.
    Yields:
        Tuple[int, float]: Iteration idx and objective function value for the best candidate
    """
    # accessing the dimensionality of the input vector
    dimensions = len(bounds)
    # initializing the population to random size
    pop = np.random.rand(population_size, dimensions)
    # obtaining the extreme values for the bounds
    min_b, max_b = np.asarray(bounds).T
    # absolute range per component in bounds
    diff = np.fabs(min_b - max_b)
    # de-normalizing (from (0-1) to another range) the population
    pop_denorm = min_b + pop * diff
    # evaluating the function at each point in the population
    fitness = np.asarray([objective_function(ind) for ind in pop_denorm])
    # obtaining the index corresponding to the minimal value of the objective function
    best_idx = np.argmin(fitness)
    # obtaining the best element in the population
    best = pop_denorm[best_idx]
    for i in range(maxit):
        # cycling over the population members
        for j in range(population_size):
            # accessing all the indexes a part from the one corresponding to the best candidate
            idxs = [idx for idx in range(population_size) if idx != j]
            # sampling without replacement three elements of the population at random to apply mutation
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            # mapping the mutant to the 0-1 range
            mutant = np.clip(a + mutation * (b - c), 0, 1)
            # masking those points to be overwritten according to recombination
            cross_points = np.random.rand(dimensions) < cross_p
            # making sure to recombine at least one element
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            # recombination step
            trial = np.where(cross_points, mutant, pop[j])
            # mapping the new trial vector to the range considered rather than 0-1
            trial_denorm = min_b + trial * diff
            # evaluating the objective function at this best point
            f = objective_function(trial_denorm)
            if f < fitness[j]: # if trial point is good, save it
                fitness[j] = f 
                pop[j] = trial
                if f < fitness[best_idx]: # if trial point is better than current best then store it
                    best_idx = j
                    best = trial_denorm
        if verbose != 0 and i % print_every == 0: 
            print(f"Iteration {i} - Objective Function value: {fitness[best_idx]}")
    return best