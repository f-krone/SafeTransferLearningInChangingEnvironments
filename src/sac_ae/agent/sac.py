import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w]
    return image


class SAC(object):
    def __init__(self, model, device, action_shape, args):
        self.model = model
        self.device = device
        self.actor_update_freq = args.actor_update_freq
        self.critic_target_update_freq = args.critic_target_update_freq
        self.critic_tau = args.critic_tau
        self.encoder_tau = args.critic_encoder_tau
        self.image_size = args.agent_image_size
        self.log_interval = args.log_interval
        self.discount = args.discount
        self.detach_encoder = args.detach_encoder
        self.robot = args.robot_shape > 0
        self.train_cost_critic = args.cost in ['critic_train', 'critic_eval']
        self.train_actor_with_cost = args.cost == 'critic_train'
        self.cost_samples = args.cost_samples
        self.cost_allowed_threshold = args.cost_allowed_threshold
        
        self.log_alpha = torch.tensor(np.log(args.init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(action_shape)
        
        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor.parameters(), lr=args.actor_lr, betas=(args.actor_beta, 0.999))

        self.critic_optimizer = torch.optim.Adam(
            self.model.critic.parameters(), lr=args.critic_lr, betas=(args.critic_beta, 0.999))

        if self.train_cost_critic:
            self.cost_critic_optimizer = torch.optim.Adam(
                self.model.cost_critic.parameters(), lr=args.critic_lr, betas=(args.critic_beta, 0.999))

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999))


        self.train()
        self.model.critic_target.train()
        if self.train_cost_critic:
            self.model.cost_critic_target.train()

    def train(self, training=True):
        self.training = training
        self.model.actor.train(training)
        self.model.critic.train(training)
        if self.train_cost_critic:
            self.model.cost_critic.train()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        if self.robot:
            if obs['image'].shape[-1] != self.image_size:
                obs['image'] = center_crop_image(obs['image'], self.image_size)
        else:
            if obs.shape[-1] != self.image_size:
                obs = center_crop_image(obs, self.image_size)
            
        with torch.no_grad():
            if self.robot:
                obs_torch = {k: torch.FloatTensor(obs[k]).to(self.device).unsqueeze(0) for k in obs.keys()}
            else:
                obs_torch = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
            mu, _, _, _ = self.model.actor(
                obs_torch, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()
    
    def select_low_cost_action(self, obs):
        if not self.train_cost_critic:
            return self.select_action(obs)

        if self.robot:
            if obs['image'].shape[-1] != self.image_size:
                obs['image'] = center_crop_image(obs['image'], self.image_size)
        else:
            if obs.shape[-1] != self.image_size:
                obs = center_crop_image(obs, self.image_size)

        with torch.no_grad():
            if self.robot:
                obs_torch = {k: torch.FloatTensor(obs[k]).to(self.device).unsqueeze(0) for k in obs.keys()}
            else:
                obs_torch = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
            # There is probably room for improvements here if passing a batch to the actor and critic, but this works for now.
            actions = []
            for _ in range(self.cost_samples):
                _, pi, _, _ = self.model.actor(obs_torch, compute_log_pi=False)
                actions.append(pi)
            costs = list(map(lambda x: max(self.model.cost_critic(obs_torch, x)), iter(actions)))
            low_cost = list(filter(lambda x: x[1] < self.cost_allowed_threshold, enumerate(costs)))
            if len(low_cost) > 0:
                action = max(map(lambda x: (min(self.model.critic(obs_torch, actions[x[0]])), actions[x[0]]), iter(low_cost)), key=lambda x: x[0])[1]
            else:
                action = actions[min(enumerate(costs), key=lambda x: x[1])[0]]
            return action.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        if self.robot:
            if obs['image'].shape[-1] != self.image_size:
                obs['image'] = center_crop_image(obs['image'], self.image_size)
        else:
            if obs.shape[-1] != self.image_size:
                obs = center_crop_image(obs, self.image_size)

        with torch.no_grad():
            if self.robot:
                obs_torch = {k: torch.FloatTensor(obs[k]).to(self.device).unsqueeze(0) for k in obs.keys()}
            else:
                obs_torch = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
            mu, pi, _, _ = self.model.actor(obs_torch, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.model.actor(next_obs)
            target_Q1, target_Q2 = self.model.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.model.critic(
            obs, action, detach=self.detach_encoder)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_cost_critic(self, obs, action, cost, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.model.actor(next_obs)
            target_Q1, target_Q2 = self.model.cost_critic_target(next_obs, policy_action)
            target_V = torch.max(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = cost + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.model.cost_critic(
            obs, action, detach=True)# Only train the encoder on the reward critic. Not sure what is best here.
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        if step % self.log_interval == 0:
            L.log('train_cost_critic/loss', critic_loss, step)

        # Optimize the critic
        self.cost_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.cost_critic_optimizer.step()

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.model.actor(obs, detach=True)

        actor_Q1, actor_Q2 = self.model.critic(obs, pi, detach=True)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        
        if self.train_actor_with_cost:
            cost_Q1, cost_Q2 = self.model.cost_critic(obs, pi, detach=True)
            cost_Q = torch.max(cost_Q1, cost_Q2)
            
            actor_loss = (self.alpha.detach() * log_pi + cost_Q - actor_Q).mean()
        else:
            actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * \
            (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:                                    
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


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



    def save_model(self, dir, step):
        torch.save(self.model.state_dict(), os.path.join(dir, f'{step}.pt'))

    def load_model(self, file, map_location=None):
        self.model.load_state_dict(torch.load(file, map_location=map_location))
    def load_model_from_dict(self, weights):
        self.model.load_state_dict(weights)
