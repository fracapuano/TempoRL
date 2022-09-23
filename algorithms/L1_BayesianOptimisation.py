""" 
This script performs Bayesian Optimisation to optimize pulse shape.

Author: Francesco Capuano, ELI-beamlines intern, Summer 2022. 
"""
# these imports are necessary to import modules from directories one level back in the folder structure
from multiprocessing.sharedctypes import Value
import sys
import os
from typing import Tuple
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from utils.losses import Losses
import numpy as np
import argparse
import matplotlib.pyplot as plt
from utils.physics import extract_data, mse
from utils import LaserModel as LM
from environment.env_utils.LaserModel_wrapper import LaserWrapper
from utils.visualize_actions import AnimateActions

from bayes_opt import BayesianOptimization

def parse_args()->None: 
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-file", default="BayesianControl.txt", type=str, help="filename to store training information")
    parser.add_argument("--controls-fname", default="Controls_BayesControl.mp4", type=str, help="filename to be used to store the visualization")
    parser.add_argument("--pulses-fname", default="Pulses_BayesControl.mp4", type=str, help="filename to be used to store the visualization")
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

def main()->None: 
    frequency, field = extract_data() # extracting the desired information
    l1_pump = model(frequency, field ** 2)
    losses = Losses(laser = l1_pump)
    # control is applied in (d2, d3, d4)
    objective_function = lambda d2, d3, d4: -1 * losses.loss6(x = np.array((d2, d3, d4)))
    
    GDDperc, TODperc, FODperc = 0.1, 0.5, 0.6
    # stretcher control bounds are centered in the compressor bounds and have a width related to a given percentage
    # (which can be though of as an hyperparameter as long as it is in the tunable interval)

    low_stretcher, high_stretcher = (
        -1 * np.array((l1_pump.compressorGDD, l1_pump.compressorTOD, l1_pump.compressorFOD)) * np.array((1 - GDDperc, 1 - TODperc, 1 - FODperc)), 
        -1 * np.array((l1_pump.compressorGDD, l1_pump.compressorTOD, l1_pump.compressorFOD)) * np.array((1 + GDDperc, 1 + TODperc, 1 + FODperc))
                                    )

    # stretcher control must be given in terms of dispersion coefficients so they must be translated into d2, d3 and d4. 
    low_stretcher, high_stretcher = (
        l1_pump.translate_control(low_stretcher, verse = "to_disp"),
        l1_pump.translate_control(high_stretcher, verse = "to_disp")
    )

    # these are the bounds for the parameter currently optimized - sign can change so sorting is used
    pbounds = {
        "d2": np.sort((low_stretcher[0], high_stretcher[0])), 
        "d3": np.sort((low_stretcher[1], high_stretcher[1])), 
        "d4": np.sort((low_stretcher[2], high_stretcher[2]))
    }

    # optimizers instantiation
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        verbose=1, # verbose=1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=10,
    )
    # these are hyperparameters of the optimization process
    n_init, n_iter = 10, 139
    optimizer.maximize(
        init_points=n_init,
        n_iter=n_iter,
    )

    performed_controls = np.zeros(shape = (len(optimizer.res)+1, 3))
    for i, res in enumerate(optimizer.res): 
        performed_controls[i,:] = np.fromiter(res["params"].values(), dtype = np.float64)
    
    # plotting the result obtained with Bayesian Optimization
    optimald2, optimald3, optimald4 = optimizer.max["params"]["d2"], optimizer.max["params"]["d3"], optimizer.max["params"]["d4"]
    performed_controls[-1,:] = np.fromiter(optimizer.max["params"].values(), dtype = np.float64)
    np.savetxt(f"../trainings/{args.training_file}", performed_controls)

    optimal_control = np.array([optimald2, optimald3, optimald4])
    # applying optimal control found
    time_BO, profile_BO = l1_pump.forward_pass(optimal_control)

    if args.render_output: # whether or not to plot the final result
        fig, ax = plt.subplots()
        ax.set_title("Pulse Optimization results", fontsize = 12, fontweight = "bold")
        
        ax.plot(time_BO, np.roll(profile_BO, - np.argmax(profile_BO) + np.argmax(losses.target_profile)), label = "Bayesian Optimization\noutput", lw = 2.5)
        ax.scatter(losses.target_time, losses.target_profile, label = "Target Pulse", c = "tab:grey", marker = "x", s = 50)

        ax.set_xlabel("Time (s)", fontsize = 12); ax.set_ylabel("Intensity", fontsize = 12)

        ax.legend(loc = "upper right", fontsize = 12, framealpha=1.)
        ax.set_xlim(left = -8e-12, right = 8e-12)
        fig.tight_layout()
    
    print("Final Loss between Control and Target: {:.4e}".format(mse(x = losses.target_profile, y = profile_BO)))

    if args.render_training: # whether or not to render the training process
        applied_controls = np.loadtxt(f"../trainings/{args.training_file}")
        applied_controls = np.array(list(map(lambda control: l1_pump.translate_control(control, verse = "to_gdd"), applied_controls)))
        
        lw = LaserWrapper()
        animator = AnimateActions(performed_actions=applied_controls, laser_wrapper=lw, last_actions=10)
        if args.render_pulses is False and args.render_controls is False: 
            raise ValueError("Rendering of pulses and controls can't be False at the same time.")
        if args.render_pulses:
            animator.animate_actions(show_target = True, save_anim = args.save_anim, fname = f"../media/{args.pulses_fname}")
        if args.render_controls:
            animator.animate_controls(save_anim = args.save_anim, fname = f"../media/{args.controls_fname}")
    
    plt.show()
    
if __name__ == "__main__": 
    main()