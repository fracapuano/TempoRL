""" 
This script reproduces a semi-physical model for a pump-laser. 
Author: Francesco Capuano, Summer 2022 S17 Intern @ ELI beam-lines, Prague.
"""
from utils.physics import *
# these imports are necessary to import modules from directories one level back in the folder structure
import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from typing import Tuple
import numpy as np
from scipy.constants import c
from numpy.fft import fft, ifft, fftfreq, fftshift

class LaserModel: 
    def __init__(
        self,
        frequency:np.array,
        intensity:np.array, 
        cutoff:Tuple[float, float], 
        num_points:int=int(5e3), 
        num_points_padding:int=int(6e4), 
        compressor_params:Tuple[float, float, float]=(1e3, 1e3, 1e3), 
        B:float=2) -> None:
        """Init function. 
        This model is initialized for a considered intensity in the frequency domain signal.

        Args:
            frequency (np.array): Array of frequencies, measured in THz.
            intensity (np.array): Array of intensity (measured with respect to the frequency).
            cutoff (Tuple[float, float]): The frequencies to be used as frequency cutoff (in Hz).
            num_points (int, optional): Number of points that need to be syntetically generated in the cutoff[0]-cutoff[1] interval.
            num_points_padding (int, optional): Number of points to be used to pad. Defaults to int(6e4)
            at a given distante equal to (abs(cutoff[1]-cutoff[0])/num_points). Defaults to int(5e3).
            compressor_params (Tuple[float, float, float]): Compressor GDD, TOD and FOD. These are considered.
            laser-characteristic and are not controlled, therefore are essentially speaking hyper-parameters to the process.
            B (float, optional): B-integral value. Used to model the non-linear effects that DIRA has on the beam.
        """
        self.frequency = frequency * 10 ** 12 # THz to Hz
        self.field = np.sqrt(intensity) # electric field is the square root of intensity
        self.preprocessed = False
        # parametrization of the preprocessing step
        self.cutoff = np.array(cutoff) * 10**12
        self.num_points = num_points
        self.pad_points = num_points_padding
        # number of points to be used in padding 
        # hyperparameters - LASER parametrization
        self.compressorGDD, self.compressorTOD, self.compressorFOD = compressor_params
        self.B = B
        
    def preprocessing(self):
        """This function applies necessary preprocessing steps to the spectrum of the laser.
        """
        if not self.preprocessed: # pre-processing only if necessary
            # cleaning the signal (cutting off the parts which are affected by measurement issues)
            self.frequency, self.field = cutoff_signal(frequency_cutoff = self.cutoff, frequency=self.frequency, signal = self.field)
            # obtaining num_points equally distant measurements
            self.frequency, self.field = equidistant_points(frequency = self.frequency, signal = self.field, num_points = self.num_points)
            # computing the central carrier of the signal
            self.central_frequency = central_frequency(frequency = self.frequency, signal = self.field)
            # preprocessing complete
            self.preprocessed = True
        else: 
            pass
    
    def stretcher(self, control:np.array) -> np.array:
        """This function imposes a phase on frequency-represented intensity according to control parameters. Control parameters relate to 
        those which actually control the phase with a system of linear equations whose solution is actually at the vector parametrizing the
        phase considered.

        Args:
            control (np.array): Control parameters for the phase.

        Raises:
            ValueError: The control parameters are three and three only, (d2, d3 and d4).

        Returns:
            np.array: The field in the laser (y1), which is the result on the phase imposition on the field entering the stretcher.
        """
        # sanity check
        if len(control) != 3: 
            raise ValueError("Controlling exactly 3 parameters: d2, d3 and d4")
        # central wavelength (using c/f = lambda, of course)
        central_wavelength = c / self.central_frequency

        d2, d3, d4 = control # the actually controlled parameters
        # linear systems of equations that links control parameters to the phase parametrization
        a11 = (-2 * np.pi * c)/(central_wavelength ** 2) #; a12 = a13 = 0
        a21 = (4 * np.pi * c)/(central_wavelength ** 3); a22 = ((2 * np.pi * c)/(central_wavelength ** 2))**2 # a23 = 0
        a31 = (-12 * np.pi * c)/(central_wavelength ** 4); a32 = (-24 * (np.pi * c) ** 2)/(central_wavelength ** 5); a33 = -((2 * np.pi * c) / (central_wavelength ** 2)) ** 3

        # solving the conversion system using forward substitution
        GDD = d2 / a11; TOD = (d3 - a21 * GDD)/(a22); FOD = (d4 - a31 * GDD - a32 * TOD)/(a33)
        # obtaining the phase 
        controlled_phase = phase_equation(frequency = self.frequency, central_frequency = self.central_frequency, GDD = GDD, TOD = TOD, FOD = FOD)
        
        # imposing the phase on the actual field considered
        return self.field * np.exp(1j * controlled_phase)

    def DIRA(self) -> np.array: 
        """This function applies a non linear phase on the signal considered according to a parametrized
        model.

        Returns:
            np.array: y2, intensity in the frequency domain considering DIRA.
        """
        y1_time = ifft(self.y1)
        intensity_time = ((y1_time*np.conj(y1_time)))**2
        intensity_0 = intensity_time.max()
        nonlinear_phase = (self.B / intensity_0) * intensity_time

        # impose the non linear phase on the electric field in time
        y1_time = y1_time * np.exp(1j * nonlinear_phase)
        return fft(y1_time)

    def compressor(self) -> np.array: 
        """This function imposes the compressor phase on the output of DIRA (stored in the y2 attribute).
        This phase is not directly controlled, even though it depends on certain values characteristic of the laser. The compressor
        phase parametrization is not controlled on-line in this model.
        
        Returns:
            np.array: y3, intensity in the frequency domain considering the compressor phase application.
        """
        compressor_phase = phase_equation(frequency = self.frequency,
                                          central_frequency = self.central_frequency,
                                          GDD = self.compressorGDD, 
                                          TOD = self.compressorTOD, 
                                          FOD = self.compressorFOD
        )
        # imposing the compressor phase on the input field
        return self.y2 * np.exp(1j * compressor_phase)
    
    def FROG(self) -> np.array: 
        time = fftshift(fftfreq(n = len(self.y3), d = np.diff(self.frequency)[0]))

        # padding the spectral intensity and phase to increase sample complexity for the fft algorithm
        field_time = fftshift(ifft(self.y3))
        
        intensity_time = np.real(field_time * np.conj(field_time)) # only for casting reasons
        intensity_time = intensity_time / intensity_time.max() # normalizing intensity

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
        self.y2 = self.DIRA()
        # obtaining y3
        self.y3 = self.compressor()
        # padding y3 with zeros on the tails (to increase fft algorithm precision)
        self.y3 = np.pad(self.y3, pad_width = (self.pad_points // 2, self.pad_points // 2), mode = "constant", constant_values = (0, 0))
        # obtaining FROG in time
        time, self.y3_time = self.FROG()
        return time, self.y3_time