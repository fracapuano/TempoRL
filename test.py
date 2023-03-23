"""
Trains a given policy on the laser environment.
"""
from env.LaserEnv_v1 import LaserEnv_v1
from env.env_utils import EnvParametrization
from utils import reverseAlgoDict
import argparse

trainsteps_dict = {
    1e4: "1e4", 
    2e5: "2e5"
}
 
def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def parse_args()->object: 
    """Args function. 
    Returns:
        (object): args parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="PPO", type=str, help="RL Algorithm. One in ['TRPO', 'PPO', 'A2C', 'SAC']")
    parser.add_argument("--verbose", default=0, type=int, help="Verbosity value")
    parser.add_argument("--train-timesteps", default=1e5, type=float, help="Number of timesteps to train the RL algorithm with")
    parser.add_argument("--test-episodes", default=50, type=int, help="Number of test matches the agent plays during periodic evaluation")
    parser.add_argument("--model-path", default=None, type=str, help="Model path")
    parser.add_argument("--render", default=False, type=boolean_string, help="Whether or not to render the environment")
    
    parser.add_argument("--default", default=True, type=boolean_string, help="Default mode, ignore all configurations")
    return parser.parse_args()

args = parse_args()

algorithm=args.algorithm
verbose=args.verbose
train_timesteps=args.train_timesteps
render=args.render
test_episodes=args.test_episodes
model_path=args.model_path

if args.default: 
    algorithm="PPO"
    verbose=2
    train_timesteps=1e4
    test_episodes=50
    render=True
    model_path="trainedmodels"

def main(): 
    # no seed is setted, but it can be easily done uncommenting the following lines
    seed = None
    # np.random.seed(seed)
    # random.seed(seed)

    checkpoint_frequency = 25_000

    if algorithm.upper() not in ["TRPO", "PPO", "A2C", "SAC"]:
        print(f"Prompted algorithm (upper): {algorithm.upper()}")
        raise ValueError("Algorithm currently supported are ['TRPO', 'PPO', 'A2C', 'SAC'] only!")

    # define version (for accessing the trained model)
    version = "v1"
    # define xi
    params = EnvParametrization()
    compressor_params, bounds, B_integral = params.get_parametrization()
    # define the environment (on top of xi)
    env = LaserEnv_v1(
        bounds = bounds, 
        compressor_params = compressor_params, 
        B_integral = B_integral, 
        render_mode="human")
    
    # retrieve the name of the model trained
    model_name = algorithm.upper() + version + "_" + trainsteps_dict[train_timesteps]
    
    model_function = reverseAlgoDict[algorithm.upper()]
    model = model_function.load(model_path + "/" + model_name + ".zip")
    # setting the name for filenames
    
    for _ in range(test_episodes): 
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, info = env.step(action)

            if render: 
                env.render()

if __name__ == "__main__": 
    main()

