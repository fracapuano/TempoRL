import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.constants import c  # speed of light

from typing import Tuple
from utils.funcs import get_project_root

def cutoff_signal(frequency_cutoff:Tuple[float, float], frequency:np.array, signal:np.array) -> Tuple[np.array, np.array]:
    """This function cuts the input signal using input frequency cutoff. It returns the cutted signal. 
    The cut is done so as to discard the part of the signal which is non-zero only for measurement noise.

    Args:
        frequency_cutoff (Tuple[float, float]): Lower and upper cutoff frequency, measured in THz.
        frequency (np.array): Array of frequencies considered, measured in THz.
        signal (np.array): The signal of interest (represented in the frequency).

    Raises:
        ValueError: Lower cutoff frequency must always be strictly smaller than upper cutoff frequency.

    Returns:
        Tuple[np.array, np.array]: Cutted arrays of frequency and field, (frequency, field). 
    """
    low_f, up_f = frequency_cutoff

    if low_f >= up_f:  # consistency
        raise ValueError("frequency_cutoff must be a tuple of strictly increasing values.")
    
    left_idx, right_idx = np.argwhere(frequency >= low_f)[0].item(), np.argwhere(frequency <= up_f)[-1].item()
    return frequency[left_idx:right_idx+1], signal[left_idx:right_idx+1]

def equidistant_points(frequency:np.array, signal:np.array, num_points:int=int(5e3)) -> Tuple[np.array, np.array]: 
    """This function uses a spline to interpolate the input signal so as to be able to reproduce it for
    equidistant frequencies rather than for the actual sampled ones (which are not equidistant) This step is
    fundamental to be able to later on apply the FFT algorithm.

    Args:
        frequency (np.array): Array of frequencies considered measured in THz.
        signal (np.array): The signal of interest (represented in the frequency domain).

    Returns:
        Tuple[np.array, np.array]: Equidistant array of frequencies, Correspondant values of electric field, (frequency, field). 
    """
    spline = UnivariateSpline(sorted(frequency), signal, s = 0)  # use all points available
    equidistant_frequency = np.linspace(start = frequency.min(), stop = frequency.max(), num = num_points)

    return (equidistant_frequency, spline(equidistant_frequency))

def extract_data(data_path:str=None)->Tuple[np.array, np.array]: 
    """This function extracts the desired information from the data file given.
    
    Args: 
        data_path (str, optional): Path where one can read experimental measurements. Defaults to None.

    Returns: 
        Tuple[np.array, np.array]: Frequency (in THz) and Field (Square-root of intensity) arrays.
    """
    if data_path is None:
        # by default, data are stored in the data folder
        data_path = str(get_project_root()) + "/data/L1_pump_spectrum.csv"

    # read the data
    df = pd.read_csv(data_path, header = None)
    df.columns = ["Wavelength (nm)", "Intensity"]
    # converting Wavelength (nm) to Frequency (THz)
    df["Frequency (THz)"] = df["Wavelength (nm)"].apply(lambda wavelenght: 1e-12 * (c/(wavelenght * 1e-9)))
    # clipping everything that is negative - measurement error
    df["Intensity"] = df["Intensity"].apply(lambda intensity: np.clip(intensity, a_min = 0, a_max = None))
    # the observations must be returned for increasing values of frequency
    df = df.sort_values(by = "Frequency (THz)")

    frequency, intensity = df.loc[:, "Frequency (THz)"].values, df.loc[:, "Intensity"].values
    # mapping intensity in the 0-1 range, as it is always done
    intensity = intensity / intensity.max()
    field = np.sqrt(intensity)
    
    return frequency, field
