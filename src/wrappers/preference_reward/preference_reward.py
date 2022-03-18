from gym import Env, Wrapper
import numpy as np
from wrappers.preference_reward.model_wrapper import ModelWrapper
import wandb

class PreferenceReward(Wrapper):
    #TODO calc max_mse from the env.action_space
    def __init__(self, env: Env, preferenceModel: ModelWrapper, max_mse: float, alpha: float, use_wandb: bool=False) -> None:
        super().__init__(env)
        self.preferenceModel = preferenceModel
        self.alpha = alpha
        self.use_wandb = use_wandb
        self.max_mse = max_mse
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

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.external_rewards.append(reward)
        internal_reward = self._calc_internal_reward(observation, action)
        rew = (1 - self.alpha) * reward - self.alpha * internal_reward
        self.calc_rewards.append(rew)
        return observation, rew, done, info

    
    def reset(self, **kwargs):
        if self.use_wandb and len(self.internal_rewards) > 0:
            wandb.log({
                "ep_internal_reward_mean": np.mean(self.internal_rewards),
                "ep_external_reward_mean": np.mean(self.external_rewards),
                "ep_confidence_mean": np.mean(self.confidences),
                "ep_action_error_mean": np.mean(self.actions_errors),
                "ep_calc_reward_mean": np.mean(self.calc_rewards)})
        self.internal_rewards = []
        self.external_rewards = []
        self.confidences = []
        self.actions_errors = []
        observation = self.env.reset(**kwargs)
        return observation
