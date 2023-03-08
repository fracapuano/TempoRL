""" 
This script performs Differential Evolution to optimize pulse shape.

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

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import Bounds
from utils.losses import Losses
from utils import LaserModel as LM
import argparse
from environment.env_utils.LaserModel_wrapper import LaserWrapper
from utils.visualize_actions import AnimateActions
from utils.physics import extract_data, mse

def parse_args()->None: 
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", default=1, type=float, help="verbose level. Any verbose > 0 prints every print-every some info")
    parser.add_argument("--print-every", default=10, type=int, help="print every printevery iterations some info")
    parser.add_argument("--maxit", default=int(3e3), type=int, help="maximal number of iterations for the algorithm")
    parser.add_argument("--store-training", default=True, type=bool, help="whether or not to store the training in a given training file")
    parser.add_argument("--training-file", default="DiffEvControl.txt", type=str, help="filename to store training information")
    parser.add_argument("--controls-fname", default="Controls_DiffEvControl.mp4", type=str, help="filename to be used to store the visualization")
    parser.add_argument("--pulses-fname", default="Pulses_DiffEvControl.mp4", type=str, help="filename to be used to store the visualization")
    parser.add_argument("--render-training", default=False, type=bool, help="Whether or not to render the training process once observed")
    parser.add_argument("--render-output", default=True, type=bool, help="Whether or not to plot the final control")
    parser.add_argument("--save-anim", default=True, type=bool, help="Whether or not to save the animation")
    parser.add_argument("--render-pulses", default=True, type=bool, help="Whether or not to render pulses observed")
    parser.add_argument("--render-controls", default=True, type=bool, help="Whether or not to render controls applied")
    return parser.parse_args()

args = parse_args()

def model(
    frequency:np.array,
    intensity:np.array, 
    compressor_params:np.array=-1 * np.array((267.422 * 1e-24, -2.384 * 1e-36, 9.54893 * 1e-50)), # in s^2, s^3 and s^4 (SI units)
    B:float=2., 
    cutoff_frequencies:Tuple=(289.95, 291.91), 
    num_points:int=int(5e3))->object: 
    """
    Returns the model instantiated (and with preprocessing already done) with respect to given arguments.
    Check LaserModel to access more pieces of documentation.
    """
    l1_pump = LM.LaserModel(
        frequency = frequency, 
        intensity = intensity,
        compressor_params = compressor_params,
        B = B, 
        cutoff = cutoff_frequencies,
        num_points=num_points)
    l1_pump.preprocessing()
    return l1_pump

def bounds(GDDperc=0.05, TODperc=0.4, FODperc=0.3): 
    frequency, field = extract_data() # extracting the desired information
    intensity = field ** 2

    l1_pump = model(frequency, intensity)
    COMPRESSOR = np.array((l1_pump.compressorGDD, l1_pump.compressorTOD, l1_pump.compressorFOD))
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
    l1_pump = model(frequency, intensity)
    COMPRESSOR = np.array((l1_pump.compressorGDD, l1_pump.compressorTOD, l1_pump.compressorFOD))

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
    print_every:int=10,
    store_training:bool=True):
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
        store_training (bool, optional): Whether or not to store the points probed during training.
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
    bests = np.zeros(shape = (maxit, dimensions))
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
        bests[i, :] = best
    # storing training
    if store_training:
        np.savetxt(f"../trainings/{args.training_file}", bests)
    return best

def main()->None: 
    # extracting data
    frequency, field = extract_data()
    # instantiating the model
    l1_pump = model(frequency=frequency, intensity = field ** 2)
    losses = Losses(laser = l1_pump)
    # selecting the loss function - check notebook to explore reasons behind this choice
    loss_function = losses.loss3
    # defining the optimization bounds
    bounds_DE = Bounds(
            # GDD         # TOD          # FOD
    lb = (2.3522e-22, -1.003635e-34, 4.774465e-50),
    ub = (2.99624e-22, 9.55955e-35, 1.4323395e-49)
    )
    # translating the control bounds to dispersion parameters
    bounds_matrix = np.vstack((bounds_DE.lb, bounds_DE.ub)).T
    disp_bounds = np.sort(l1_pump.translate_control(bounds_matrix, verse = "to_disp"))
    # optimizing using diff evolution
    best_DE = diff_evolution(
        objective_function=loss_function, 
        bounds=disp_bounds, 
        verbose=args.verbose, 
        maxit=args.maxit,
        store_training=args.store_training)
    
    time_DE, profile_DE = l1_pump.forward_pass(best_DE)
    
    if args.render_output: # whether or not to plot the final result
        fig, ax = plt.subplots()
        ax.set_title("Pulse Optimization results", fontsize = 12, fontweight = "bold")
        
        ax.plot(time_DE, np.roll(profile_DE, - np.argmax(profile_DE) + np.argmax(losses.target_profile)), label = "Diff. Evolution\noutput", lw = 2.5)
        ax.scatter(losses.target_time, losses.target_profile, label = "Target Pulse", c = "tab:grey", marker = "x", s = 50)

        ax.set_xlabel("Time (s)", fontsize = 12); ax.set_ylabel("Intensity", fontsize = 12)

        ax.legend(loc = "upper right", fontsize = 12, framealpha=1.)
        ax.set_xlim(left = -8e-12, right = 8e-12)
        fig.tight_layout()
    
    print("Final Loss between Control and Target: {:.4e}".format(mse(x = losses.target_profile, y = profile_DE)))

    if args.render_training: # whether or not to render the training process
        applied_controls = np.loadtxt(f"../trainings/{args.training_file}")
        applied_controls[-1,:] =  best_DE
        # translating the control parameters from dispersion to GDD, TOD and FOD. 
        applied_controls = np.array(list(map(lambda control: l1_pump.translate_control(control, verse = "to_gdd"), applied_controls)))
        lw = LaserWrapper()
        animator = AnimateActions(performed_actions=applied_controls, laser_wrapper=lw, last_actions=5)
        if args.render_pulses is False and args.render_controls is False: 
            raise ValueError("Rendering of pulses and controls can't be False at the same time.")
        if args.render_pulses:
            animator.animate_actions(show_target = True, save_anim = args.save_anim, fname = f"../media/{args.pulses_fname}")
        if args.render_controls:
            animator.animate_controls(save_anim = args.save_anim, fname = f"../media/{args.controls_fname}")
    
    plt.show()

if __name__=="__main__": 
    main()
