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