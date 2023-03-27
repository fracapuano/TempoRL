import numpy as np
import wandb
from typing import Tuple
from env.BaseLaser import Abstract_BaseLaser

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.evaluation import evaluate_policy

class StatsAgg:
    def __init__(self): 
        self.reset_stats()

    def lasertraining_callback(self, locals_dict:dict, globals_dict:dict):
        """
        Uses access to locals() to elaborate information. 
        Intended to be used inside of stable_baselines3 `evaluate_policy`
        """
        if locals_dict["done"]:
            self.loss_stoppages += locals_dict["info"].get("LossStoppage", False)
            self.timesteps_stoppages += locals_dict["info"].get("TimeStepsStoppage", False)
            self.duration_stoppages += locals_dict["info"].get("DurationStoppage", False)
            self.final_FWHM.append(locals_dict["info"].get("current FWHM (ps)", 0))
            self.final_Intensity.append(locals_dict["info"].get("current Peak Intensity", 0))
            self.episode_lens.append(locals_dict["info"]["episode"]["l"])
            self.terminal_loss.append(locals_dict["info"]["L1Loss"])
    
    def reset_stats(self):
        self.loss_stoppages = 0
        self.timesteps_stoppages = 0
        self.duration_stoppages = 0
        self.final_FWHM = []
        self.final_Intensity = []
        self.episode_lens = []
        self.terminal_loss = []
        
class PulseTrainingCallback(BaseCallback): 
    """Custom callback inheriting from `BaseCallback`.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug.

    Performs various actions when triggered (intended to be a child of EventCallback): 
        1. Evaluates current policy (for n_eval_episodes)
        2. Updates a current best_policy variable
        3. Logs stuff on wandb. More details on what is logged in :meth:_on_step.
    """
    def __init__(
            self, 
            env:Tuple[Abstract_BaseLaser, VecEnv], 
            render:bool=False, 
            verbose:int=0,
            n_eval_episodes:int=50, 
            best_model_path:str="models/"):
        """Init function defines callback context."""
        super().__init__(verbose)

        self._envs = env
        self.render = render
        self.n_eval_episodes = n_eval_episodes
        self.EvaluationStats = StatsAgg()
        # resets environment
        self._envs.reset()
        # current best model and best model's return in test trajectories
        self.best_model_path = best_model_path
        self.best_model = None
        self.best_model_mean_reward = -np.inf
        self.bests_found = 0

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `_env.step()`.
        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        # flush past statistics
        self.EvaluationStats.reset_stats()
        # obtain mean and std of cumulative reward over n_eval_episodes
        mean_cum_reward, std_cum_reward = evaluate_policy(
            self.model,
            self.model.get_env(),
            callback=self.EvaluationStats.lasertraining_callback,
            n_eval_episodes=self.n_eval_episodes, 
            render=self.render)
        std_cum_reward = std_cum_reward if std_cum_reward > 0 else 1e-6
        
        wandb.log({
            "(%) LossStoppage": round(self.EvaluationStats.loss_stoppages / self.n_eval_episodes, 2),
            "(%) TimeStepsStoppage": round(self.EvaluationStats.timesteps_stoppages / self.n_eval_episodes, 2),
            "Avg(EpisodeLen)": np.mean(self.EvaluationStats.episode_lens),
            "Min(TerminalState-L1Loss)": np.min(self.EvaluationStats.terminal_loss),
            "Max(TerminalState-L1Loss)": np.max(self.EvaluationStats.terminal_loss),
            "Avg(TerminalState-L1Loss)": np.mean(self.EvaluationStats.terminal_loss), 
            "Std(TerminalState-L1Loss)": np.std(self.EvaluationStats.terminal_loss)
        })
        
        # checks if this model is better than current best. If so, update current best
        if mean_cum_reward >= self.best_model_mean_reward:
            self.best_model = self.model
            self.best_model_mean_reward = mean_cum_reward
            # save best model
            self.best_model.save(path=f"{self.best_model_path}/best_model.zip")
            self.bests_found += 1

        wandb.log({
             "Mean Cumulative Reward": mean_cum_reward, 
             "Std of Cumulative Reward": std_cum_reward,
        })

        return True
    
    def get_best_model(self, return_reward:bool=True): 
        if return_reward:
            return self.best_model, self.best_model_mean_reward
        else: 
            return self.best_model

class IntensityTrainingCallback(BaseCallback): 
    """Custom callback inheriting from `BaseCallback`.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug.

    Performs various actions when triggered (intended to be a child of EventCallback): 
        1. Evaluates current policy (for n_eval_episodes)
        2. Updates a current best_policy variable
        3. Logs stuff on wandb. More details on what is logged in :meth:_on_step.
    """
    def __init__(
            self, 
            env:Tuple[Abstract_BaseLaser, VecEnv], 
            render:bool=False, 
            verbose:int=0,
            n_eval_episodes:int=50, 
            best_model_path:str="models/"):
        """Init function defines callback context."""
        super().__init__(verbose)

        self._envs = env
        self.render = render
        self.n_eval_episodes = n_eval_episodes
        self.EvaluationStats = StatsAgg()
        # resets environment
        self._envs.reset()
        # current best model and best model's return in test trajectories
        self.best_model_path = best_model_path
        self.best_model = None
        self.best_model_mean_reward = -np.inf
        self.bests_found = 0

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `_env.step()`.
        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        # flush past statistics
        self.EvaluationStats.reset_stats()
        # obtain mean and std of cumulative reward over n_eval_episodes
        mean_cum_reward, std_cum_reward = evaluate_policy(
            self.model, 
            self.model.get_env(), 
            callback=self.EvaluationStats.lasertraining_callback,
            n_eval_episodes=self.n_eval_episodes, 
            render=self.render)
        std_cum_reward = std_cum_reward if std_cum_reward > 0 else 1e-6
        
        wandb.log({
            "(%) TimeStepsStoppage": round(self.EvaluationStats.timesteps_stoppages / self.n_eval_episodes, 2),
            "Avg(EpisodeLen)": np.mean(self.EvaluationStats.episode_lens),
            "Avg(TerminalState-L1Loss)": np.mean(self.EvaluationStats.terminal_loss),
            "Min(Final-FWHM)": np.min(self.EvaluationStats.final_FWHM), 
            "Max(Final-FWHM)": np.max(self.EvaluationStats.final_FWHM),
            "Avg(Final-FWHM)": np.mean(self.EvaluationStats.final_FWHM),
            "Std(Final-FWHM)": np.std(self.EvaluationStats.final_FWHM),
            "Min(Final-PeakIntensity)": np.min(self.EvaluationStats.final_Intensity), 
            "Max(Final-PeakIntensity)": np.max(self.EvaluationStats.final_Intensity),
            "Avg(Final-PeakIntensity)": np.mean(self.EvaluationStats.final_Intensity),
            "Std(Final-PeakIntensity)": np.std(self.EvaluationStats.final_Intensity),
        })
        
        # checks if this model is better than current best. If so, update current best
        if mean_cum_reward / std_cum_reward >= self.best_model_reward_over_std:
            self.best_model = self.model
            self.best_model_mean_reward = mean_cum_reward
            # save best model
            self.best_model.save(path=f"{self.best_model_path}/best_model.zip")
            self.bests_found += 1

        wandb.log({
             "Mean Cumulative Reward": mean_cum_reward,
             "Std of Cumulative Reward": std_cum_reward,
        })
        return True
    
    def get_best_model(self, return_reward:bool=True): 
        if return_reward:
            return self.best_model, self.best_model_mean_reward
        else: 
            return self.best_model

