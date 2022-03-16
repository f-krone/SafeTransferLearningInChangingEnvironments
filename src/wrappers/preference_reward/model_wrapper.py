from typing import List, Tuple
from stable_baselines3 import SAC
from stable_baselines3.common.base_class import BaseAlgorithm
import numpy as np

class ModelWrapper:
    def __init__(self, models: List[BaseAlgorithm]) -> None:
        self.models = models

    def __init__(self, sac_model_paths: List[str]) -> None:
        self.models = list(map(lambda path: SAC.load(path), iter(sac_model_paths)))

    def calc_action(self, observation) -> Tuple[np.ndarray, float]:
        actions = np.expand_dims(list(map(lambda model: model.predict(observation)[0], iter(self.models))), 0)
        action = actions.mean(axis=1)
        confidence = np.exp(-actions.std(axis=1).mean())
        return action, confidence