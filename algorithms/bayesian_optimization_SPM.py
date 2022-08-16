""" 
This script performs Bayesian Optimisation to optimize pulse shape.

Author: Francesco Capuano, ELI-beamlines intern, Summer 2022. 
"""
# these imports are necessary to import modules from directories one level back in the folder structure
import sys
import os
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from utils import physics, LaserModel as LM

def model(frequency, intensity, compressor, b_int, cutoff)->object: 
    """
    Returns the model instantiated with respect to arguments. 
    Consistency of arguments type is assumed in this function.
    """
    l1_pump = LM.LaserModel(
        frequency = frequency, 
        intensity = intensity,
        compressor_params = compressor, 
        B = b_int, 
        cutoff = cutoff)
    
    return l1_pump

def main()->None: 
    filename = "LLNL_160809_freq.csv"
    df = pd.read_csv("data/"+filename, header = None)
    df.columns = ["Frequency (in THz)", "Wavelength (in nm)", "Intensity", "Phase (rad)", "Phase (cutted) (rad)"] 
    df = df.sort_values(by = "Frequency (in THz)")

    COMPRESSOR, B, CUTOFF = np.array((20, 30, 40)) * 1e3, 2, (350, 380)
    frequency = df.loc[:, "Frequency (in THz)"].values # in THz
    intensity = np.sqrt(df.loc[:, "Intensity"].values)

    l1_pump = model(frequency, intensity, COMPRESSOR, B, CUTOFF)

    
    

    
    plt.show()
if __name__ == "__main__": 
    main()

    
