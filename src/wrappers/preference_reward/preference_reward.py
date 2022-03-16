from gym import Env, Wrapper
from wrappers.preference_reward.model_wrapper import ModelWrapper
import wandb

class PreferenceReward(Wrapper):
    def __init__(self, env: Env, preferenceModel: ModelWrapper, alpha: float, use_wandb: bool=False) -> None:
        super().__init__(env)
        self.preferenceModel = preferenceModel
        self.internal_reward = 0
        self.alpha = alpha
        self.use_wandb = use_wandb

    def reward(self, reward: float) -> float:
        if self.use_wandb:
            wandb.log({"internal_reward": self.internal_reward, "external_reward": reward}, commit=False)
        return reward - self.alpha * self.internal_reward

    def _calc_internal_reward(self, observation, action):
        preference_action, confidence = self.preferenceModel.calc_action(observation)
        action_error = ((action - preference_action)**2).mean()
        self.internal_reward = confidence * action_error

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._calc_internal_reward(observation, action)
        return observation, self.reward(reward), done, info

    
    def reset(self, **kwargs):
        wandb.log({})
        observation = self.env.reset(**kwargs)
        return observation
