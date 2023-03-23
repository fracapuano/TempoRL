"""
Trains a given policy on the laser environment.
"""
from env.LaserEnv_v1 import LaserEnv_v1
from env.env_utils import EnvParametrization
from utils import reverseAlgoDict

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import wandb
from wandb.integration.sb3 import WandbCallback
import argparse

trainsteps_dict = {
    1e4: "1e4", 
    2e5: "2e5", 
    5e5: "5e5"
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
    parser.add_argument("--evaluate_while_training", default=True, type=boolean_string, help="Whether or not to evaluate the RL algorithm while training")
    parser.add_argument("--store-checkpoints", default=True, type = boolean_string, help="Whether or not to store partially-trained models. Recommended True for long trainings (>1e6 ts)")
    parser.add_argument("--evaluation-frequency", default=1e3, type = float, help="Frequency with which to evaluate policy against random fair opponent")
    parser.add_argument("--test-episodes", default=50, type=int, help="Number of test matches the agent plays during periodic evaluation")
    parser.add_argument("--save-model", default=False, type=boolean_string, help="Whether or not save the model currently trained")
    parser.add_argument("--resume-training", default=False, type=boolean_string, help="Whether or not load and keep train an already trained model")
    parser.add_argument("--model-path", default=None, type=str, help="Path to which the model to incrementally train is stored")

    parser.add_argument("--default", default=True, type=boolean_string, help="Default mode, ignore all configurations")
    return parser.parse_args()

args = parse_args()

algorithm=args.algorithm
verbose=args.verbose
train_timesteps=args.train_timesteps
evaluate_while_training=args.evaluate_while_training
store_checkpoints=args.store_checkpoints
evaluation_frequency=args.evaluation_frequency
test_episodes=args.test_episodes
save_model=args.save_model
resume_training=args.resume_training
model_path=args.model_path

if args.default: 
    algorithm="PPO"
    verbose=0
    train_timesteps=1e4
    evaluate_while_training=True
    store_checkpoints=True
    evaluation_frequency=1000
    test_episodes=50
    save_model=True
    resume_training=False
    model_path=None

def main(): 
    # no seed is setted, but it can be easily done uncommenting the following lines
    seed = None
    # np.random.seed(seed)
    # random.seed(seed)

    checkpoint_frequency = 25_000

    if algorithm.upper() not in ["TRPO", "PPO", "A2C", "SAC"]:
        print(f"Prompted algorithm (upper): {algorithm.upper()}")
        raise ValueError("Algorithm currently supported are ['TRPO', 'PPO', 'A2C', 'SAC'] only!")

    # define version (for logging)
    version = "v1"

    training_config = {
    "version": version,
    "model": algorithm.upper(),
    "total_timesteps": train_timesteps,
    }
    
    # using wandb to log the training process
    if True:
        run = wandb.init(
        project="DeepPulse-SPIE",
        config=training_config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True
        )
        # using W&B during training
        wand_callback = WandbCallback(verbose=2, gradient_save_freq=500)

    # define xi
    params = EnvParametrization()
    compressor_params, bounds, B_integral = params.get_parametrization()
    # define the environment (on top of xi)
    def make_env():
        env = LaserEnv_v1(
            bounds = bounds, 
            compressor_params = compressor_params, 
            B_integral = B_integral)
        env = Monitor(env)
        return env
    
    # vectorized environment, wrapped with video recorder
    env = DummyVecEnv([make_env])

    # retrieve the algorithm used
    model_function = reverseAlgoDict[algorithm.upper()]
    model = model_function(
        "MlpPolicy",
        env=env, 
        verbose=verbose, 
        seed=seed,
        tensorboard_log=f"runs/{run.id}"
    )
    # setting the name for filenames
    model_name = algorithm.upper() + version + "_" + trainsteps_dict[train_timesteps]
    # saving a model every tot timesteps
    checkpoint_save = CheckpointCallback(
        save_freq=evaluation_frequency, save_path="checkpoints/", name_prefix=f"{algorithm}"
    )
    # list of callbacks (then, DR should be implemented in here)
    callback_list = [checkpoint_save]
    # adding wandb callback to other callbacks
    if True:
        callback_list.append(wand_callback)

    # training the model with train_timesteps
    if not resume_training:
        print(f"Training: {model_name}")
        model.learn(total_timesteps=train_timesteps, callback=callback_list, progress_bar=True)
    
    else:
        raise NotImplementedError("Incremental learning has not been implemented yet!")
        # only resuming training of partially-trained model
        # remaining_training_steps = int(float(input("Please enter new number of training steps: ")))

    if save_model: 
        model.save(f"trainedmodels/{model_name}.zip")

if __name__ == "__main__": 
    main()

