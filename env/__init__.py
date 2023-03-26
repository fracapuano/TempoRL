from .LaserModel import *
from .BaseLaser import *
from .LaserEnv_v1 import *

from .env_utils import *
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from .LaserEnv_v1 import LaserEnv_v1
from stable_baselines3.common.monitor import Monitor

VersionDict = dict(
    v1=LaserEnv_v1,
)

def build_default_env(
        version:str="v1", 
        n_envs:int=1, 
        subprocess:bool=True, 
        device="cpu",
        **kwargs)->VecEnv:
    """Simply builds an env using default configuration for the environment.

    Args: 
        n_envs (int, optional): Number of different envs to instantiate (using VecEnvs). Defaults to 1.
        subprocess (bool, optional): Whether or not to create multiple copies of the same environment using 
                                     suprocesses (so that multiple envs can run in parallel). When False, uses
                                     the usual DummyVecEnv. Defaults to True.
    
    Returns: 
    """    
    # define xi
    params = EnvParametrization()
    compressor_params, bounds, B_integral = params.get_parametrization()
    # define environment (on top of xi)
    def make_env():
        env = VersionDict[version](
            # first four arguments are always the same for all LaserEnv(s)
            bounds,
            compressor_params, 
            B_integral, 
            device
            # kwargs differentiate between different versions
            **kwargs)
        # wrapping using monitor
        env = Monitor(env)
        return env

    # vectorized environment, wrapped with Monitor
    if subprocess:
        env = SubprocVecEnv([make_env for _ in range(n_envs)], start_method="fork")  # maybe change, test on heph
    else: 
        env = DummyVecEnv([make_env for _ in range(n_envs)])

    return env