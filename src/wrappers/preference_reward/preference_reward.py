from typing import Callable, Union
from gym import Env, Wrapper
import numpy as np
from wrappers.preference_reward.model_wrapper import ModelWrapper
from torch.utils.tensorboard import SummaryWriter

class PreferenceReward(Wrapper):
    #TODO calc max_mse from the env.action_space
    def __init__(self, env: Env, preferenceModel: ModelWrapper, max_mse: float, alpha: Union[int, Callable[[float], float], str], tensorboard_log: str=None, logger=None) -> None:
        super().__init__(env)
        self.preferenceModel = preferenceModel
        self.alpha = alpha
        self.tensorboard_log = tensorboard_log
        self.max_mse = max_mse
        self.steps = 0
        if tensorboard_log != None:
            self.writer = SummaryWriter(log_dir=tensorboard_log)
        else:
            self.writer = None
        self.logger = logger
        self.env_reward_mean = 0
        self.internal_rewards = []
        self.external_rewards = []
        self.confidences = []
        self.actions_errors = []
        self.calc_rewards = []

    def _calc_internal_reward(self, observation, action) -> float:
        preference_action, confidence = self.preferenceModel.calc_action(observation)
        action_error = ((action - preference_action)**2).mean()
        internal_reward = confidence * action_error / self.max_mse
        self.internal_rewards.append(internal_reward)
        self.confidences.append(confidence)
        self.actions_errors.append(action_error)
        return internal_reward

    def _get_alpha(self) -> float:
        if callable(self.alpha):
            return self.alpha(self.steps)
        if type(self.alpha) == str and self.alpha == 'auto':
            assert self.env_reward_mean <= 0, "Env reward is always below tero, the mean should be as well"
            #return np.min([1.0, 1.0-1.05**(self.env_reward_mean)])
            reward_min = -15
            reward_max = 0
            alpha = 1.0 - (reward_min - self.env_reward_mean) / (reward_min - reward_max)
            alpha = np.max([np.min([1.0, alpha]), 0.0])
            assert alpha >= 0.0, "Alpha should be at least 0"
            assert alpha <= 1.0, "Alpha should be at most 1"
            return alpha
        return self.alpha

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.steps += 1
        self.external_rewards.append(reward)
        internal_reward = self._calc_internal_reward(observation, action)
        rew = (1 - self._get_alpha()) * reward - self._get_alpha() * internal_reward
        self.calc_rewards.append(rew)
        return observation, rew, done, info

    
    def reset(self, **kwargs):
        if self.env_reward_mean == 0:
            self.env_reward_mean = np.sum(self.external_rewards)
        else:
            self.env_reward_mean = 0.01 * np.sum(self.external_rewards) + 0.99 * self.env_reward_mean
        if self.writer != None and len(self.internal_rewards) > 0:
            self.writer.add_scalar('preference_reward/ep_internal_reward_mean', np.mean(self.internal_rewards))
            self.writer.add_scalar('preference_reward/ep_external_reward_mean', np.mean(self.external_rewards))
            self.writer.add_scalar('preference_reward/ep_confidence_mean', np.mean(self.confidences))
            self.writer.add_scalar('preference_reward/ep_action_error_mean', np.mean(self.actions_errors))
            self.writer.add_scalar('preference_reward/ep_calc_reward_mean', np.mean(self.calc_rewards))
            self.writer.add_scalar('preference_reward/alpha', self._get_alpha())
            self.writer.add_scalar('preference_reward/env_reward_moving_avg', np.mean(self.env_reward_mean))
        if self.logger != None and len(self.internal_rewards) > 0:
            self.logger.log('train/preference_reward/ep_internal_reward_mean', np.mean(self.internal_rewards), self.steps)
            self.logger.log('train/preference_reward/ep_external_reward_mean', np.mean(self.external_rewards), self.steps)
            self.logger.log('train/preference_reward/ep_confidence_mean', np.mean(self.confidences), self.steps)
            self.logger.log('train/preference_reward/ep_action_error_mean', np.mean(self.actions_errors), self.steps)
            self.logger.log('train/preference_reward/ep_calc_reward_mean', np.mean(self.calc_rewards), self.steps)
            self.logger.log('train/preference_reward/alpha', self._get_alpha(), self.steps)
            self.logger.log('train/preference_reward/env_reward_moving_avg', np.mean(self.env_reward_mean), self.steps)
        self.internal_rewards = []
        self.external_rewards = []
        self.confidences = []
        self.actions_errors = []
        observation = self.env.reset(**kwargs)
        if self.preferenceModel.teacher_per_episode:
            self.preferenceModel.select_teacher()
        return observation
