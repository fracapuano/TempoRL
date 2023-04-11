"""
Trains a given policy on the laser environment.
"""
from env import get_default_env
from policy.policy import Policy
import numpy as np
import argparse
from rich.progress import track

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
    parser.add_argument("--algorithm", default="PPO", type=str, help="RL Algorithm. One in ['TRPO', 'PPO', 'SAC']")
    parser.add_argument("--env-version", default="v1", type=str, help="Version of custom env to use. One in ['v1', 'v2']")
    parser.add_argument("--verbose", default=0, type=int, help="Verbosity value")
    parser.add_argument("--test-episodes", default=50, type=int, help="Number of test matches the agent plays during periodic evaluation")
    parser.add_argument("--model-path", default=None, type=str, help="Model path")
    parser.add_argument("--render", action="store_true", help="Whether or not to render the environment")
    parser.add_argument("--seed", default=777, type=int, help="Random seed")
    
    parser.add_argument("--default", action="store_true", help="Default mode, ignore all configurations")
    return parser.parse_args()

args = parse_args()

algorithm=args.algorithm
env_version=args.env_version
verbose=args.verbose
render=args.render
test_episodes=args.test_episodes
model_path=args.model_path
render=args.render
seed=args.seed

if args.default: 
    algorithm="PPO"
    verbose=1
    test_episodes=50
    render=True
    model_path="models/earthy-jazz-6_models/best_model.zip"

def main(): 
    # build the envs according to spec
    env = get_default_env(
        version=env_version, 
        render_mode="human" if render else "rgb_array"
    )
    # if rendering, increase fps
    if render: 
        env.metadata["render_fps"] = 15
        
    # create policy - loading pretrained model
    policy = Policy(
        algo=algorithm,
        env=env,
        seed=seed, 
        load_from_pathname=model_path) 
    
    episodes_rewards = np.zeros(test_episodes)
    for ep in track(range(test_episodes), description="Testing episodes..."):
        episode_rewards = []
        episode_losses=[]
        obs = env.reset()
        done = False
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            episode_rewards.append(reward)
            episode_losses.append(info["L1Loss"])
            
            if render:
                env.render()
        
        episodes_rewards[ep] = np.mean(episode_rewards)
    
    if verbose: 
        print(f"Avg-Cumulative reward over {test_episodes}: {np.mean(episodes_rewards)}")

if __name__ == "__main__": 
    main()

