""" 
This script implements the Losses class, used to flexibly define new loss function to carry out pulse optimisation.

Author: Francesco Capuano, ELI-beamlines intern, Summer 2022. 
"""
# these imports are necessary to import modules from directories one level back in the folder structure
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

def model(frequency:np.array, intensity:np.array, compressor:np.array, b_int:int, cutoff:Tuple[float, float], num_points:int)->object: 
    """
    Returns the model instantiated with respect to arguments. 
    To access full documentation check LaserModel docstring.
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

class Losses: 
    def __init__(self, laser:object): 
        """Init function.

        Args:
            laser (object): LaserModel object - Used to retrieve all possible information on the signal with ease. 
        """
        self.laser = laser
        # pre-processed version of frequency and intensity
        frequency_clean, field_clean = self.laser.spit_center()

        self.target_time, self.target_profile = physics.temporal_profile(
            frequency_clean, 
            physics.amplification(frequency_clean, field_clean),
            phase = np.zeros_like(frequency_clean), 
            npoints_pad = self.laser.pad_points)

        self.tol = 1e-6 # zero tolerance
    
    def loss1(self, x:np.array,)->float: 
        """This function implements the first loss, i.e. a weighted squared error where the weights are the intensity values considered.

        Args:
            x (np.array): Control quantities 

        Returns:
            float: Error measure
        """        
        controlled_profile = self.laser.forward_pass(x)[1]
        controlled_profile = np.roll(controlled_profile, np.argmax(controlled_profile) - np.argmax(self.target_profile))
        return (controlled_profile * (controlled_profile - self.target_profile)**2).sum() / controlled_profile.sum()

    def loss2(self, x:np.array) -> float: 
        """This function implements the second loss, i.e. a MSE masked to the portion of the array where the controlled temporal profile is above a certain
        value of threshold. 
        
        Args:
            x (np.array): Control quantities 

        Returns:
            float: Error measure
        """   
        controlled_profile = self.laser.forward_pass(x)[1]
        mask = (controlled_profile != self.target_profile) & (controlled_profile > self.tol)
        
        controlled_profile = controlled_profile[mask]
        target = self.target_profile[mask]
        
        return ((controlled_profile - target)**2).mean()
    
    def loss3(self, x:np.array) -> float: 
        """This function implements the third loss, i.e. a MSE masked to the portion of the array where the controlled temporal profile is above a certain
        value of threshold, combined with a distance measure related to the difference between the area underlined by the two pulses. 
        
        Args:
            x (np.array): Control quantities 

        Returns:
            float: Error measure
        """
        w1, w2 = 0.3, 0.7
        controlled_profile = self.laser.forward_pass(x)[1]
        mask = (controlled_profile != self.target_profile) & (controlled_profile > self.tol)
        
        controlled_profile = controlled_profile[mask]
        target = self.target_profile[mask]
        
        return (w1 * ((controlled_profile - target)**2).mean() + w2 * (np.trapz(controlled_profile) - np.trapz(target)))

    def loss4(self, x:np.array) -> float: 
        """This function implements the fourth loss, i.e. a MSE masked to the portion of the array where the controlled temporal profile is above a certain
        value of threshold, combined with the sum of the widths of all the peaks present in the signal (this is done so as to reduce as much as possible
        the width of FWHM while still considering the other peaks eventually present in the signal itself). 
        
        Args:
            x (np.array): Control quantities 

        Returns:
            float: Error measure
        """
        w1, w2 = 0.3, 0.7
        controlled_profile = self.laser.forward_pass(x)[1]
        mask = controlled_profile > self.tol
        
        controlled_profile = controlled_profile[mask]
        target = self.target_profile[mask]
        
        control_peaks, _ = find_peaks(controlled_profile, height = (0.1, None))
        target_peaks, _ = find_peaks(target, height = (0.1, None))
        
        control_width = peak_widths(controlled_profile, control_peaks)[0]
        target_width = peak_widths(target, target_peaks)[0]
        
        mse_part = ((controlled_profile - target)**2).mean() 
        peak_part = (target_width.sum() - control_width.sum())**2
        return w1 * mse_part + w2 * peak_part
    
    def loss5(self, x:np.array) -> float: 
        """This function implements the fifth loss, i.e. a MSE masked to the portion of the array where the controlled temporal profile is above a certain
        value of threshold, combined with the difference between the target FWHM and the controlled one. 
        
        Args:
            x (np.array): Control quantities 

        Returns:
            float: Error measure
        """
        w1, w2 = 0.1, 0.9
        controlled_time, controlled_profile = self.laser.forward_pass(x)
        
        control_roots = UnivariateSpline(
            x = controlled_time, y = controlled_profile - (controlled_profile.max()/2), s = 0
        ).roots()
            
        control_FWHM = np.abs(control_roots[0] - control_roots[-1])
        target_FWHM = np.diff(UnivariateSpline(
            x = self.target_time, y = self.target_profile - (self.target_profile.max()/2), s = 0
        ).roots())
        
        mask = controlled_profile > self.tol

        controlled_profile = controlled_profile[mask]
        target = self.target_profile[mask]
        
        shape_part = ((controlled_profile - target)**2).mean()
        FWHM_part = np.abs((control_FWHM - target_FWHM).item())
        
        return w1 * shape_part + w2 * FWHM_part
    
    def loss6(self, x:np.array) -> float: 
        """This function implements the sixth loss, i.e. a L1-Manhattan norm masked to the portion of the array where the controlled temporal profile is above a certain
        value of threshold. 

        Args:
            x (np.array): Control quantities 

        Returns:
            float: Error measure
        """
        controlled_profile = self.laser.forward_pass(x)[1]
        mask = (controlled_profile > self.tol)
        
        controlled_profile = controlled_profile[mask]
        target = self.target_profile[mask]
        
        return (np.abs(controlled_profile - target)).sum()
 