from policy.policy import Policy
from policy.callbacks import PulseTrainingCallback
from env import build_default_env

envs = build_default_env(n_envs=3, subprocess=True)
policy = Policy(algo="ppo", env=envs)
policy.train(
    timesteps=5, callback_list=[PulseTrainingCallback(env=envs, n_eval_episodes=1)]
)
