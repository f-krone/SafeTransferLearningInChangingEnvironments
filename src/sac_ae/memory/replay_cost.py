import traceback
from collections import defaultdict
import numpy as np
import torch

from .replay_drq import ReplayBuffer, ReplayBufferDataset, ReplayBufferStorage, episode_len


class ReplayBufferStorageCost(ReplayBufferStorage):
    def __init__(self, replay_dir, robot=False, state=False):
        super(ReplayBufferStorageCost, self).__init__(replay_dir, robot, state)

    def add(self, state, action, reward, cost, done):
        if state is not None:
            self._current_episode['s'].append(state)
        if action is not None:
            self._current_episode['a'].append(action)
        if reward is not None:
            self._current_episode['r'].append(reward)
        if cost is not None:
            self._current_episode['c'].append(cost)
        
        if done:
            episode = dict()
            episode['s'] = np.array(self._current_episode['s'], dict if self.robot else (np.float32 if self.state else np.uint8))
            episode['a'] = np.array(self._current_episode['a'], np.float32)
            episode['r'] = np.array(self._current_episode['r'], np.float32)
            episode['c'] = np.array(self._current_episode['c'], np.float32)

            self._current_episode = defaultdict(list)
            self._store_episode(episode)

class ReplayBufferDatasetCost(ReplayBufferDataset):
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot):
        super(ReplayBufferDatasetCost, self).__init__(replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot)
    
    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1)
        obs = episode['s'][idx]
        action = episode['a'][idx]
        next_obs = episode['s'][idx + self._nstep]
        reward = np.zeros_like(episode['r'][idx])
        cost = np.zeros_like(episode['c'][idx])
        discount = 1
        for i in range(self._nstep):
            step_reward = episode['r'][idx]
            step_cost = episode['c'][idx]
            reward += discount * step_reward
            cost += discount * step_cost
            discount *= self._discount
        return (obs, action, reward, cost, next_obs)

class ReplayBufferCost(ReplayBuffer):
    def __init__(self, iter, obs_shape, device, image_size=84, image_pad=None, robot=False):
        super(ReplayBufferCost, self).__init__(iter, obs_shape, device, image_size, image_pad, robot)

    def sample(self):
        (obs, action, reward, cost, next_obs) = next(self.iter)
        reward = torch.unsqueeze(reward, dim=-1)
        cost = torch.unsqueeze(cost, dim=-1)
        not_done = torch.ones_like(reward)  # episode ends because of maximal step limits 
        if self.robot:
            obs['image'] = obs['image'].float().to(self.device)
            obs['robot'] = obs['robot'].float().to(self.device)
            next_obs['image'] = next_obs['image'].float().to(self.device)
            next_obs['robot'] = next_obs['robot'].float().to(self.device)
        else:
            obs = obs.float().to(self.device)
            next_obs = next_obs.float().to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        cost = cost.to(self.device)
        not_done = not_done.to(self.device)
        
        return obs, action, reward, cost, next_obs, not_done