"""
This script implements basic functions useful for the software engineering part of this project.
"""
from pathlib import Path
import json
from typing import Tuple
import numpy as np
import pandas as pd
from scipy.constants import c

def get_project_root() -> Path:
    """
    Returns always project root directory.
    """
    return Path(__file__).parent.parent

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
    intensity = intensity / intensity.max()
    field = np.sqrt(intensity)
    
    return frequency, field

def renumber_cells(path:str) -> None:
    """This function refactors a ipynb file so to have subsequent values.

    Args: 
        path (str): path where ipynb file is located with respect to parent directory. 
    Returns: 
        None: (file with increasing cell number)
    """
    if not path.endswith(".ipynb"): 
        raise ValueError("Script defined for notebooks only. Insert extension / change file type to notebook")
    NOTEBOOK_FILE = str(get_project_root()) + "/" + path
    with open(NOTEBOOK_FILE, 'rt') as f_in:
        doc = json.load(f_in)
    cnt = 1
    for cell in doc['cells']:
        if 'execution_count' not in cell:
            continue
        cell['execution_count'] = cnt
        for o in cell.get('outputs', []):
            if 'execution_count' in o:
                o['execution_count'] = cnt
        cnt = cnt + 1
    with open(NOTEBOOK_FILE, 'wt') as f_out:
        json.dump(doc, f_out, indent=1)

    print("done")