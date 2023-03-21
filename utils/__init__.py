from .funcs import *
from .physics import *
from .preprocessing import *
from .render import * 
from .torch_utils import *

from stable_baselines3 import PPO, SAC, A2C
from sb3_contrib import TRPO


# dictionary to test out diffrent algorithms
reverseAlgoDict = {
    "PPO": PPO, 
    "SAC": SAC,
    "A2C": A2C, 
    "TRPO": TRPO
}