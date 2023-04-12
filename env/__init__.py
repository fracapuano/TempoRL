from .LaserModel import *
from .BaseLaser import *
from .LaserEnv_v1 import *

from .env_utils import *
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from .LaserEnv_v1 import LaserEnv_v1
from .LaserEnv_v2 import LaserEnv_v2
from stable_baselines3.common.monitor import Monitor

VersionDict = dict(
    v1=LaserEnv_v1,
    v2=LaserEnv_v2
)

def build_default_env(
        version:str="v1", 
        n_envs:int=1, 
        subprocess:bool=True, 
        device="cpu",
        render_mode="rgb_array",
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
            bounds=bounds,
            compressor_params=compressor_params, 
            B_integral=B_integral, 
            render_mode=render_mode,
            device=device,
            # kwargs differentiate between different versions
            env_kwargs=kwargs)
        # wrapping using monitor
        env = Monitor(env)
        return env

    # vectorized environment, wrapped with Monitor
    if subprocess:
        env = SubprocVecEnv([make_env for _ in range(n_envs)], start_method="fork")  # maybe change, test on heph
    else: 
        env = DummyVecEnv([make_env for _ in range(n_envs)])

    return env

def get_default_env(
        version:str="v1",
        device:str="cpu",
        render_mode:str="rgb_array",
        init_variance:float=.1,
        **kwargs)->Abstract_BaseLaser:
    """Simply builds an env using default configuration for the environment.

    Args: 
        version (str, optional): Environment version. Used to distinguish between version1 and version2. Defaults to "v1".
        device (str, optional): Device on which to run the environment. Defaults to "cpu".
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
    
    Returns: 
        Abstract_BaseLaser: gym.Env of the laser considered.
    """    
    # define xi
    params = EnvParametrization()
    compressor_params, bounds, B_integral = params.get_parametrization()
    # define environment (on top of xi)
    env = VersionDict[version](
        # first four arguments are always the same for all LaserEnv(s)
        bounds=bounds,
        compressor_params=compressor_params, 
        B_integral=B_integral, 
        render_mode=render_mode,
        device=device,
        init_variance=init_variance,
        # kwargs differentiate between different versions
        env_kwargs=kwargs)
        # wrapping using monitor
    
    return env
