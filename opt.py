from env.LaserEnv_v1 import LaserEnv_v1
from env.env_utils import EnvParametrization

params = EnvParametrization()
compressor_params, bounds, B_integral = params.get_parametrization()

env = LaserEnv_v1(
    bounds = bounds, 
    compressor_params = compressor_params, 
    B_integral = B_integral)

print(env._get_info())