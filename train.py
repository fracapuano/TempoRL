from policy.policy import Policy
from policy.callbacks import PulseTrainingCallback, IntensityTrainingCallback
from stable_baselines3.common.callbacks import EveryNTimesteps
import wandb
import argparse
from env import build_default_env
from utils.funcs import to_scientific_notation

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
    parser.add_argument("--algorithm", default="PPO", type=str, help="RL Algorithm. One in ['TRPO', 'PPO', 'SAC']")
    parser.add_argument("--env-version", default="v1", type=str, help="Version of custom env to use. One in [v1, v2]")
    parser.add_argument("--verbose", default=0, type=int, help="Verbosity value")
    parser.add_argument("--train-timesteps", default=1e5, type=float, help="Number of timesteps to train the RL algorithm with")
    parser.add_argument("--evaluation-frequency", default=1e4, type = float, help="Frequency with which to evaluate policy against random fair opponent")
    parser.add_argument("--test-episodes", default=25, type=int, help="Number of test matches the agent plays during periodic evaluation")
    parser.add_argument("--resume-training", action="store_true", help="Whether or not load and keep train an already trained model")
    parser.add_argument("--model-path", default="models/", type=str, help="Path to which the model to incrementally train is stored")
    parser.add_argument("--render", action="store_true", help="Boolean flag related to whether or not to render the env")
    parser.add_argument("--seed", default=777, type=int, help="Random seed setted")
    parser.add_argument("--gamma", default=0.9, type=float, help="Discount factor")
    parser.add_argument("--increment", default=False, type=boolean_string, help="Use actual value or difference in values for different envs")
    parser.add_argument("--healthy-coeff", default=0.1, type=float, help="Healthy-reward coefficientf")
    parser.add_argument("--performance-coeff", default=1., type=float, help="Performance reward coefficient")
    parser.add_argument("--n-envs", default=1, type=int, help="Number of different envs to create at training time")

    parser.add_argument("--default", action="store_true", help="Default mode, ignore all configurations")
    return parser.parse_args()

args = parse_args()

algorithm=args.algorithm
env_version=args.env_version
verbose=args.verbose
train_timesteps=args.train_timesteps
evaluate_every=args.evaluation_frequency
test_episodes=args.test_episodes
resume_training=args.resume_training
model_path=args.model_path
render=args.render
seed=args.seed
GAMMA=args.gamma
INC_IMPROVEMENT=args.increment
COEFFS=[args.healthy_coeff, args.performance_coeff]
n_envs=args.n_envs

if args.default: 
    algorithm="PPO"
    env_version="v1"
    train_timesteps=2e5
    test_episodes=25
    evaluate_every=1e4

def main():
    """Performs training and logs info to wandb."""
    run_custom_name = False
    env_args = {
        "reward_coeff": COEFFS, 
        "incremental_improvement": INC_IMPROVEMENT
    }
    # build the envs according to spec
    envs = build_default_env(
        version=env_version, 
        n_envs=n_envs, 
        subprocess=True, 
        **env_args)

    # training config dictionary
    training_config = dict(
        algorithm=algorithm,
        env_version=env_version,
        discount_factor=GAMMA,
        train_timesteps=train_timesteps,
        random_seed=seed,
        upper_bound=envs.get_attr("MAX_LOSS")[0] if env_version=="v1" else envs.get_attr("MAX_DURATION")[0],
        max_timesteps=envs.get_attr("MAX_STEPS")[0],
        coefficients=COEFFS, 
        incremental_improvement=INC_IMPROVEMENT
    )

    if verbose > 0: 
        print(training_config)

    # init wandb run
    default_name = f"{algorithm.upper()}{env_version}_{to_scientific_notation(train_timesteps)}"
    run = wandb.init(
        project=f"DeepPulse_{env_version}-Final",
        config=training_config,
        name= default_name if run_custom_name else None
        )

    # analysing the training process with a custom callback, varying
    # with respect to the environment version
    if env_version == "v1":
        pulse_callback = PulseTrainingCallback(
            env=envs, 
            render=render, 
            n_eval_episodes=test_episodes, 
            best_model_path=f"{run.name}_models/"
            )
    elif env_version == "v2":
        pulse_callback = IntensityTrainingCallback(
            env=envs, 
            render=render, 
            n_eval_episodes=test_episodes, 
            best_model_path=f"{run.name}_models/"
            )
    # invoke pulse_callback every `evaluate_every` timesteps
    evaluation_callback = EveryNTimesteps(n_steps=evaluate_every, callback=pulse_callback)
    # create policy
    policy = Policy(
        algo=algorithm,
        env=envs,
        gamma=GAMMA,
        seed=seed, 
        load_from_pathname=model_path if resume_training else None)
        
    print(f"Starting to train: {algorithm.upper()} on LaserEnv_{env_version} for {to_scientific_notation(train_timesteps)} timesteps.")
    # train policy using evaluation callback
    avg_return, std_return = policy.train(
        timesteps=train_timesteps,
        n_eval_episodes=test_episodes,
        callback_list=[evaluation_callback],
        return_best_model=False
    )
    # print the number of times a better env is found
    if verbose > 0: 
        print("BestsFound: ", evaluation_callback.callback.bests_found)
        print(f"Training completed! Best models available at: {run.name}_models/")
        print(f"Avg Return over test episodes: {round(avg_return, 2)} ± {round(std_return, 2)}")

if __name__=="__main__":
    main()
