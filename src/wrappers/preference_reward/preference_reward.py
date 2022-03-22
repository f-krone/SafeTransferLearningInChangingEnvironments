from typing import Callable, Union
from gym import Env, Wrapper
import numpy as np
from wrappers.preference_reward.model_wrapper import ModelWrapper
from torch.utils.tensorboard import SummaryWriter

class PreferenceReward(Wrapper):
    #TODO calc max_mse from the env.action_space
    def __init__(self, env: Env, preferenceModel: ModelWrapper, max_mse: float, alpha: Union[int, Callable[[float], float]], tensorboard_log: str=None) -> None:
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
        if isinstance(self.alpha, float):
            return self.alpha
        return self.alpha(self.steps)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.steps += 1
        self.external_rewards.append(reward)
        internal_reward = self._calc_internal_reward(observation, action)
        rew = (1 - self._get_alpha()) * reward - self._get_alpha() * internal_reward
        self.calc_rewards.append(rew)
        return observation, rew, done, info

    
    def reset(self, **kwargs):
        if self.writer != None and len(self.internal_rewards) > 0:
            self.writer.add_scalar('preference_reward/ep_internal_reward_mean', np.mean(self.internal_rewards))
            self.writer.add_scalar('preference_reward/ep_external_reward_mean', np.mean(self.external_rewards))
            self.writer.add_scalar('preference_reward/ep_confidence_mean', np.mean(self.confidences))
            self.writer.add_scalar('preference_reward/ep_action_error_mean', np.mean(self.actions_errors))
            self.writer.add_scalar('preference_reward/ep_calc_reward_mean', np.mean(self.calc_rewards))
            self.writer.add_scalar('preference_reward/alpha', np.mean(self._get_alpha()))
        self.internal_rewards = []
        self.external_rewards = []
        self.confidences = []
        self.actions_errors = []
        observation = self.env.reset(**kwargs)
        return observation
