""" 
This script reproduces a semi-physical model for L1 pump-laser. 
Author: Francesco Capuano, Summer 2022 S17 Intern @ ELI beamlines, Prague.
"""
import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from typing import Tuple

import torch
import numpy as np
import pandas as pd
from scipy.constants import c

from utils import physics as pt
from utils import *

cuda_available = torch.cuda.is_available()

class ComputationalLaser: 
    def __init__(
                self, 
                frequency:torch.tensor, 
                field:torch.tensor, 
                compressor_params:Tuple[float, float, float],
                num_points_padding:int=int(3e4), 
                B:float=2, 
                central_frequency:float=(c/(1030*1e-9)), 
                cristal_frequency:torch.tensor=None, 
                cristal_intensity:torch.tensor=None
                )->None:
        """Init function. 
        This model is initialized for a considered intensity in the frequency domain signal. The signal is assumed to be already cleaned. 

        Args:
            frequency (torch.tensor): Tensor of frequencies, measured in THz, already preprocessed.
            field (torch.tensor): Tensor of electrical field (measured with respect to the frequency).
            compressor_params (Tuple[float, float, float]): Compressor GDD, TOD and FOD. These are considered
                                                            laser-characteristic and are not controlled. They do parametrize the laser.
            central_frequency (float, optional): Central frequency, may be derived from central wavelength. Defaults to (c/1030*1e-9) Hz.
            num_points_padding (int, optional): Number of points to be used to pad. Defaults to int(6e4)
            B (float, optional): B-integral value. Used to model the non-linear effects that DIRA has on the beam.
            cristal_frequency (torch.tensor, optional): Frequency (THz) of the amplification in the non-linear cristal at the beginning of DIRA. Defaults to None.
            cristal_intensity (torch.tensor, optional): Intensity of the amplification in the non-linear cristal at the beginning of DIRA. Defaults to None.
        """
        self.frequency = frequency * 10 ** 12  # THz to Hz
        self.field = field  # electric field is the square root of intensity
        self.central_frequency = central_frequency
        # number of points to be used in padding 
        self.pad_points = num_points_padding
        # hyperparameters - LASER parametrization
        self.compressor_params = compressor_params
        self.B = B
        # storing the original input
        self.input_frequency, self.input_field = frequency * 1e12, field
        # YB:Yab gain
        if cristal_intensity is not None and cristal_frequency is not None: 
            self.yb_frequency = cristal_frequency * 1e12 # THz to Hz
            self.yb_field = torch.sqrt(cristal_intensity)
        else: 
            # reading the data with which to amplify the signal when non specific one is given
            cristal_path = str(get_project_root()) + "/data/cristal_gain.txt"
            gain_df = pd.read_csv(cristal_path, sep = "  ", skiprows = 2, header = None, engine = "python")

            gain_df.columns = ["Wavelength (nm)", "Intensity"]
            gain_df["Intensity"] = gain_df["Intensity"] / gain_df["Intensity"].values.max()
            gain_df["Frequency (THz)"] = gain_df["Wavelength (nm)"].apply(lambda wl: 1e12 * (c/((wl+1) * 1e-9)))  # 1nm rightwards shift

            gain_df.sort_values(by = "Frequency (THz)", inplace = True)
            yb_frequency, yb_field = gain_df["Frequency (THz)"].values, np.sqrt(gain_df["Intensity"].values)
            
            # cutting the gain frequency accordingly

            yb_frequency, yb_field = pt.cutoff_signal(
                frequency_cutoff=(self.frequency[0].item(), self.frequency[-1].item()), 
                frequency = yb_frequency, 
                signal = yb_field)
            
            # augmenting the cutted data
            yb_frequency, yb_field = pt.equidistant_points(
                frequency = yb_frequency, 
                signal = yb_field, 
                num_points = len(self.frequency)
            )
            self.yb_frequency = torch.from_numpy(yb_frequency)
            self.yb_intensity = torch.from_numpy(yb_field ** 2)
            self.yb_field = torch.from_numpy(yb_field)

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
            control (torch.tensor): Control parameters to be used to create the phase. 
                                    It contains GDD, TOD and FOD in s^2, s^3 and s^4.

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
        step = torch.diff(self.frequency)[0]
        Dt = 1 / step
        time = torch.linspace(start = - Dt/2, end = Dt/2, steps = len(self.frequency) + self.pad_points)
        # transform limited of amplified spectrum
        field_padded = torch.nn.functional.pad(
            pt.yb_gain(self.field, torch.sqrt(self.yb_intensity)),
            pad=(self.pad_points // 2, self.pad_points // 2), 
            mode = "constant", 
            value = 0
        )

        field_padded = field_padded.to("cuda") if cuda_available else field_padded

        # inverse FFT to go from frequency domain to temporal domain
        field_time = torch.fft.ifftshift(torch.fft.ifft(field_padded))
        intensity_time = torch.real(field_time * torch.conj(field_time)) # only for casting reasons

        intensity_time =  intensity_time / intensity_time.max() # normalizing
        
        # either returning time or not according to return_time
        if not return_time: 
            return intensity_time if cuda_available else intensity_time.cpu()
        else: 
            return time, intensity_time if cuda_available else intensity_time.cpu()
        
    def forward_pass(self, control:torch.tensor)->Tuple[torch.tensor, torch.tensor]: 
        """This function performs a forward pass in the model using control values stored in control.

        Args:
            control (torch.tensor): Control values to use in the forward pass. Must be dispersion coefficients. 

        Returns:
            Tuple[torch.tensor, torch.tensor]: (Time scale, Temporal profile of intensity for the given control).
        """
        # control quantities regulate the phase
        phi_stretcher = self.emit_phase(control = control)
        # phase imposed on the input field
        y1_frequency = pt.impose_phase(spectrum = self.field, phase = phi_stretcher)
        # spectrum amplified by DIRA cristal
        y1tilde_frequency = pt.yb_gain(signal = y1_frequency, intensity_yb=self.yb_field)
        # spectrum amplified in time domain, to apply non linear phase to it
        y1tilde_time = torch.fft.ifft(y1tilde_frequency)
        # defining non-linear DIRA phase
        intensity = torch.real(y1tilde_time * torch.conj(y1tilde_time))
        phi_DIRA = (self.B / intensity.max()) * intensity
        # applying non-linear DIRA phase to the spectrum
        y2_time = pt.impose_phase(spectrum = y1tilde_time, phase = phi_DIRA)
        # back to frequency domain
        y2_frequency = torch.fft.fft(y2_time)
        # defining phase imposed by compressor
        phi_compressor = self.emit_phase(control = self.compressor_params)
        # imposing compressor phase on spectrum
        y3_frequency = pt.impose_phase(y2_frequency, phase = phi_compressor)
        # return time scale and temporal profile of the (controlled) pulse
        return pt.temporal_profile(frequency = self.frequency, field = y3_frequency, npoints_pad = self.pad_points)