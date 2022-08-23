""" 
This script reproduces a semi-physical model for a pump-laser. 
Author: Francesco Capuano, Summer 2022 S17 Intern @ ELI beam-lines, Prague.
"""
from utils import physics_torch as pt
# these imports are necessary to import modules from directories one level back in the folder structure
import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from typing import Tuple
import torch
import numpy as np
from scipy.constants import c
import pandas as pd
from numpy.fft import fft, ifft, fftfreq, fftshift

class Computational_Laser: 
    def __init__(
                self, frequency:torch.tensor, intensity:torch.tensor, compressor_params:Tuple[float, float, float],
                num_points_padding:int=int(3e4), B:float=2, central_frequency:float=(c/(1030*1e-9)), 
                cristal_frequency:torch.tensor=None, cristal_intensity:torch.tensor=None
                ) -> None:
        """Init function. 
        This model is initialized for a considered intensity in the frequency domain signal. The signal is assumed to be already cleaned. 

        Args:
            frequency (torch.tensor): Tensor of frequencies, measured in THz, already preprocessed.
            intensity (torch.tensor): Tensor of intensities (measured with respect to the frequency).
            compressor_params (Tuple[float, float, float]): Compressor GDD, TOD and FOD. These are considered
                laser-characteristic and are not controlled, therefore are essentially speaking hyper-parameters to the process.
            central_frequency (float, optional): Central frequency, may be derived from central wavelength. Defaults to (c/1030*1e-9) Hz.
            num_points_padding (int, optional): Number of points to be used to pad. Defaults to int(6e4)
            B (float, optional): B-integral value. Used to model the non-linear effects that DIRA has on the beam.
            cristal_frequency (torch.tensor, optional): Frequency (THz) of the amplification in the non-linear cristal at the beginning of DIRA. Defaults to None.
            cristal_intensity (torch.tensor, optional): Intensity of the amplification in the non-linear cristal at the beginning of DIRA. Defaults to None.
        """
        self.frequency = frequency * 10 ** 12 # THz to Hz
        self.field = np.sqrt(intensity) # electric field is the square root of intensity
        self.central_frequency = central_frequency
        # number of points to be used in padding 
        self.pad_points = num_points_padding
        # hyperparameters - LASER parametrization
        self.compressorGDD, self.compressorTOD, self.compressorFOD = compressor_params
        self.B = B
        # storing the original input
        self.input_frequency, self.input_field = frequency * 10 ** 12, torch.sqrt(intensity)
        # YB:Yab gain
        if cristal_intensity is not None and cristal_frequency is not None: 
            self.yb_frequency = cristal_frequency * 1e12 # THz to Hz
            self.yb_field = torch.sqrt(cristal_intensity)
        else: 
            # reading the data with which to amplify the signal when non specific one is given
            cristal_path = "../data/cristal_gain.txt"
            gain_df = pd.read_csv(cristal_path, sep = "  ", skiprows = 2, header = None, engine = "python")

            gain_df.columns = ["Wavelength (nm)", "Intensity"]
            gain_df.Intensity = gain_df.Intensity / gain_df.Intensity.values.max()
            gain_df["Frequency (THz)"] = gain_df["Wavelength (nm)"].apply(lambda wl: 1e12 * (c/((wl+1) * 1e-9))) # 1nm shift

            gain_df.sort_values(by = "Frequency (THz)", inplace = True)
            self.yb_frequency, self.yb_field = torch.from_numpy(gain_df["Frequency (THz)"].values), torch.from_numpy(np.sqrt(gain_df["Intensity"].values))
        
    def translate_control(self, control:torch.tensor, verse:str="to_gdd")->torch.tensor: 
        """This function translates the control quantities either from Dispersion coefficients (the di's) to GDD, TOD and FOD using a system of linear equations 
        defined for this very scope or the other way around, according to the string "verse".  

        Args:
            control (torch.tensor): Control quanitities (either the di's or delay information). Must be given in SI units.
            verse (str, optional): "to_gdd" to translate control from dispersion coefficients to (GDD, TOD and FOD), solving Ax = b.
            "to_disp" to translate (GDD, TOD and FOD) to dispersion coefficient left-multiplying the control by A. Defaults to "to_gdd". 

        Returns:
            torch.tensor: The control translated according to the verse considered.
        """
        return pt.translate_control(central_frequency = self.central_frequency, control = control, verse = verse)
    
    def emit_phase(self, control:torch.tensor)->torch.tensor: 
        """This function returns the phase with respect to the frequency and some control parameters.

        Args:
            control (torch.tensor): Control parameters to be used to create the phase. It contains GDD, TOD and FOD in s^2, s^3 and s^4.

        Returns:
            torch.tensor: The phase with respect to the frequency, measured in radiants.
        """
        return pt.phase_equation(frequency = self.frequency, central_frequency = self.central_frequency, control = control)
    
    def transform_limited(self, return_time:bool=True)->Tuple[torch.tensor, torch.tensor]: 
        """This function returns the transform limited of the input spectrum.

        Args:
            return_time (bool, optional): Whether or not to return (also) the time-scale. Defaults to True.

        Returns:
            Tuple[torch.tensor, torch.tensor]: Returns either (time, intensity) (with time measured in in femtoseconds) or intensity only.
        """
        step = np.diff(self.frequency)[0]
        time = fftshift(fftfreq(len(self.frequency) + self.pad_points, d=abs(step)))

        field_padded = torch.nn.functional.pad(self.field, pad=(self.pad_points // 2, self.pad_points // 2), mode = "constant", value = 0)
        # inverse FFT to go from frequency domain to temporal domain
        field_time = torch.fft.fftshift(torch.fft.ifft(field_padded))
        intensity_time = field_time * torch.conj(field_time) # only for casting reasons

        intensity_time =  torch.real(intensity_time / intensity_time.max()) # normalizing
        
        # either returning time or not according to return_time
        if not return_time: 
            return intensity_time
        else: 
            return time, intensity_time
        
    def forward_pass(self, control:np.array) -> np.array: 
        """This function performs a forward pass in the model using control values stored in control.

        Args:
            control (np.array): Control values to use in the forward pass.

        Returns:
            np.array: Temporal profile of intensity for the given control.
        """
        # performing preprocessing operations
        self.preprocessing()
        # obtaining y1
        self.y1 = self.stretcher(control = control)
        # obtaining y2 
        if self.B != 0: # linear effect
            self.y1 = self.amplification()
            self.y2 = self.DIRA()
        else: 
            self.y2 = self.y1 # no non-linear effect
        # obtaining y3
        self.y3 = self.compressor()
        # padding y3 with zeros on the tails (to increase fft algorithm precision)
        self.y3 = np.pad(self.y3, pad_width = (self.pad_points // 2, self.pad_points // 2), mode = "constant", constant_values = (0, 0))
        # obtaining FROG in time
        time, self.y3_time = self.FROG()
        return time, self.y3_time