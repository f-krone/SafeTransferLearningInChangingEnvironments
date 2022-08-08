import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from .sac import SAC

def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs

def weight_red_channel_mean(x, weight):
    weights = torch.zeros_like(x)
    weights[:, 0::3] = weight
    weights[:, 1::3] = (3 - weight) / 2
    weights[:, 2::3] = (3 - weight) / 2
    return (weights * x).mean()


class SACAE(SAC):
    def __init__(self, model, device, action_shape, args):
        super().__init__(model, device, action_shape, args)
        
        self.autoencoder_update_freq = args.sacae_update_freq
        self.encoder_tau = args.sacae_encoder_tau
        self.robot = args.robot_shape > 0
        self.red_weight = args.sacae_red_weight
        self.cnn_3dconv = args.cnn_3dconv

        self.autoencoder_optimizer = torch.optim.Adam(
            self.model.autoencoder.parameters(), lr=args.sacae_autoencoder_lr, betas=(args.sacae_autoencoder_beta, 0.999))

        self.train()

    def train(self, training=True):
        self.training = training
        self.model.actor.train(training)
        self.model.critic.train(training)
        self.model.autoencoder.train(training)
        if self.train_cost_critic:
            self.model.cost_critic.train()


    def update_autoencoder(self, x, L, step):
        x = x['image'] if self.robot else x
        recon_x = self.model.autoencoder.recon(x)
        target = preprocess_obs(x)

        if self.red_weight == None:
            recon_loss = F.mse_loss(recon_x, target)
        else:
            recon_loss = F.mse_loss(recon_x, target, reduction="none")
            recon_loss = weight_red_channel_mean(recon_loss, self.red_weight)

        self.autoencoder_optimizer.zero_grad()
        recon_loss.backward()
        self.autoencoder_optimizer.step()
        
        if step % self.log_interval == 0:
            L.log('train/autoencoder_loss', recon_loss, step)


    def update(self, replay_buffer, L, step):
        if self.train_cost_critic:
            obs, action, reward, cost, next_obs, not_done = replay_buffer.sample()
            if step % self.log_interval == 0:
                L.log('train/batch_cost', cost.mean(), step)
            self.update_cost_critic(obs, action, cost, next_obs, not_done, L, step)
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample()
    
        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.model.soft_update_params(self.critic_tau, self.encoder_tau)
        
        if step % self.autoencoder_update_freq == 0:
            self.update_autoencoder(obs, L, step)
