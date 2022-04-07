from typing import List, Tuple
from stable_baselines3 import SAC
from stable_baselines3.common.base_class import BaseAlgorithm
import numpy as np
import wandb

class ModelWrapper:
    def __init__(self, models: List[BaseAlgorithm], obs_keys:List[str], use_mean=False) -> None:
        self.models = models
        self.use_mean = use_mean
        self.obs_keys = obs_keys

    """def __init__(self, sac_model_paths: List[str], obs_keys:List[str], use_mean=False) -> None:
        self.models = list(map(lambda path: SAC.load(path), iter(sac_model_paths)))
        self.use_mean = use_mean
        self.obs_keys = obs_keys"""

    def calc_action(self, observation) -> Tuple[np.ndarray, float]:
        # actions.shape = (len(self.models), action_space)
        actions = np.asarray(list(map(lambda model: model.predict(self._transform_obs(observation))[0], iter(self.models))))
        action = np.zeros(actions.shape[1:])
        if self.use_mean:
            action = actions.mean(axis=0)
        else:
            selected_action = np.random.randint(0, len(self.models))
            action = actions[selected_action]
        confidence = np.exp(-actions.std(axis=0).mean())
        return action, confidence
        
    def _transform_obs(self, observation):
        return {k:observation[k] for k in self.obs_keys if k in observation}