from typing import List, Tuple
from sac_ae.agent import SAC
import numpy as np
import wandb

class SACAEModelWrapper:
    def __init__(self, models: List[SAC], use_mean=False, remove_barrier=False, teacher_per_episode=False, stochastic=False) -> None:
        self.models = models
        self.use_mean = use_mean
        self.remove_barrier = remove_barrier
        self.teacher_per_episode = teacher_per_episode
        self.teacher = 0
        if stochastic:
            self.predict_func = lambda model, obs: model.sample_action(obs)
        else:
            self.predict_func = lambda model, obs: model.select_action(obs)

    def calc_action(self, observation) -> Tuple[np.ndarray, float]:
        actions = np.asarray(list(map(lambda model: self.predict_func(model, self._transform_obs(observation)), iter(self.models))))
        action = np.zeros(actions.shape[1:])
        if self.use_mean:
            action = actions.mean(axis=0)
        else:
            selected_action = self.teacher if self.teacher_per_episode else np.random.randint(0, len(self.models))
            action = actions[selected_action]
        confidence = np.exp(-actions.std(axis=0).mean())
        return action, confidence

    def select_teacher(self):
        self.teacher = np.random.randint(0, len(self.models))
        
    def _transform_obs(self, observation):
        if self.remove_barrier:
            obs = observation['observation'][:-6]
        else:
            obs = observation['observation']
        return np.concatenate([obs, observation['achieved_goal'], observation['desired_goal']])