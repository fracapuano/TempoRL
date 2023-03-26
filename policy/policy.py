"""
	Wrapper class for policy training and evaluation. 
    To be used to test different algorithms and training procedures.
    TODO: Add documentation to functions.
"""
import torch
import os

from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO, SAC
from sb3_contrib import TRPO
from typing import List

class Policy:
    def __init__(self,
                 algo=None,
                 env=None,
                 lr=3e-4,
                 device='cpu',
                 seed=None,
                 gamma=0.99,
                 load_from_pathname=None):

        if not isinstance(env, VecEnv): 
            raise ValueError(f"Env {env} is not a VecEnv!")
        # else: env = make_vec_env(env, n_envs=1, seed=seed, vec_env_cls=DummyVecEnv)

        self.seed = seed
        self.device = device
        self.env = env
        self.algo = algo.lower()
        self.gamma = gamma

        # either train from scratch (create_model) or from partially trained agent (load_model)
        if load_from_pathname is None:
            self.model = self.create_model(self.algo, lr=lr)
            self.model_loaded = False
        else:
            self.model = self.load_model(self.algo, load_from_pathname)
            self.model_loaded = True

    def create_model(self, algo, lr):
        if algo == 'ppo':
            model = PPO("MlpPolicy", 
                        self.env, 
                        learning_rate=lr,
                        seed=self.seed, 
                        device=self.device, 
                        gamma=self.gamma)

        elif algo == 'sac':
            model = SAC("MlpPolicy", 
                        self.env, 
                        learning_rate=lr,
                        seed=self.seed, 
                        device=self.device, 
                        gamma=self.gamma)
        
        elif algo == 'trpo': 
            model = TRPO("MlpPolicy", 
                        self.env, 
                        learning_rate=lr,
                        seed=self.seed, 
                        device=self.device,
                        gamma=self.gamma)
        else:
            raise ValueError(f"RL Algo not supported: {algo}. Supported algorithms ['trpo', 'ppo', 'sac']")
        return model

    def load_model(self, algo, pathname):
        if algo == 'ppo':
            model = PPO.load(pathname, env=self.env, device=self.device)
        elif algo == 'trpo':
            model = TRPO.load(pathname, env=self.env, device=self.device)
        elif algo == 'sac':
            model = SAC.load(pathname, env=self.env, device=self.device)
        else:
            raise ValueError(f"RL Algo not supported: {algo}. Supported algorithms ['trpo', 'ppo', 'sac']")
        return model

    def train(self,
              timesteps:int=1000,
              n_eval_episodes:int=50,
              show_progressbar:bool=True,
              callback_list:List[BaseCallback]=None,
              best_model_save_path:str="models/",
              return_best_model:bool=True,
              verbose:int=0, 
              reset_timesteps:bool=True):
        """
        Train a model using a custom list of callbacks. 
        Optionally, find best model among those checkpointed and returns it.
        """
        # get current verbosity level
        old_verbose = self.model.verbose
        # set verbose
        self.model.verbose = verbose
        # learns using custom callback
        self.model.learn(
            total_timesteps=timesteps, 
            callback=callback_list,
            reset_num_timesteps=reset_timesteps and self.model_loaded,
            progress_bar=show_progressbar
            )
        # reset actual verbosity level
        self.model.verbose = old_verbose

        if return_best_model:   # Find best model among last and best
            reward_final, std_reward_final = self.eval(
                n_eval_episodes=n_eval_episodes)
            
            if not os.path.exists(os.path.join(best_model_save_path, "best_model.zip")):
                # the best model is created with at every call of EventCallback (called every EventCallback.n_steps timesteps)
                # since the first check is always met (the condition is trivially n>-inf), a best_model.zip does always exist
                # when the callback has been called at least once during training!
                print("best_model.zip hasn't been saved because too few evaluations have been performed.")
                raise ValueError("Check eval_freq and training timesteps used!")
            
            best_model = self.load_model(self.algo, os.path.join(best_model_save_path, "best_model.zip"))
            
            reward_best, std_reward_best = evaluate_policy(best_model, 
                                                           best_model.get_env(),
                                                           n_eval_episodes=n_eval_episodes)
            
            std_reward_best = std_reward_best if std_reward_best > 0 else 1e-6

            # comparing std-scaled average rewards over n_episodes
            final_better_than_last = (reward_final / std_reward_final) > (reward_best / std_reward_best)
            
            if final_better_than_last: 
                best_policy = self.state_dict()
                best_mean_reward, best_std_reward = reward_final, std_reward_final
                which_one = 'final'
            else:
                best_policy = best_model.policy.state_dict()
                best_mean_reward, best_std_reward = reward_best, std_reward_best
                which_one = 'best'

            info = {'which_one': which_one}

            return best_mean_reward, best_std_reward, best_policy, info
        else:
            return self.eval(n_eval_episodes)

    def eval(self, n_eval_episodes=50, render=False):
        mean_reward, std_reward = evaluate_policy(self.model, 
                                                  self.model.get_env(), 
                                                  n_eval_episodes=n_eval_episodes, 
                                                  render=render)
        
        std_reward = std_reward if std_reward > 0 else 1e-6
        return mean_reward, std_reward

    def predict(self, state, deterministic=False):
        return self.model.predict(state, deterministic=deterministic)

    def state_dict(self):
        return self.model.policy.state_dict()

    def save_state_dict(self, pathname):
        torch.save(self.state_dict(), pathname)

    def load_state_dict(self, path_or_state_dict):
        if type(path_or_state_dict) is str:
            self.model.policy.load_state_dict(torch.load(
                path_or_state_dict, map_location=torch.device(self.device)), strict=True)
        else:
            self.model.policy.load_state_dict(path_or_state_dict, strict=True)

    def save_full_state(self, pathname):
        self.model.save(pathname)

    def load_full_state(self):
        raise ValueError('Use the constructor with load_from_pathname parameter')
        
