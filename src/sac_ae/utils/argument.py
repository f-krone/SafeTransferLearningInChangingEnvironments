from typing import List
import numpy as np
import torch
import argparse
import os
import math
import gym
import sys
import random
import time
import json
import dmc2gym
import copy


def parse_args(argumentString = None):
    parser = argparse.ArgumentParser()
    ##### Common #####
    # environment
    parser.add_argument('--env_name', default='CustomFetchPushDense-v0')
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--cost', default='no_cost', type=str)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=300000, type=int)
    # train
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=2000000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default='1024,512,256', type=str)
    # eval
    parser.add_argument('--eval_freq', default=10000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-5, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float) 
    parser.add_argument('--critic_encoder_tau', default=0.05, type=float) 
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-5, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)

    ##### Algorithm-Specific Parameters
    parser.add_argument('--agent', default='sacae', type=str, help='curl, sacae, sac, rad, drq, atc')
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--cnn_stride', default=1, type=int)
    parser.add_argument('--cnn_3dconv', default=False, action='store_true')

    # curl
    parser.add_argument('--curl_update_freq', default=1, type=int)
    parser.add_argument('--curl_lr', default=1e-3, type=float)
    parser.add_argument('--curl_encoder_tau', default=0.05, type=float)

    # sac_ae
    parser.add_argument('--sacae_update_freq', default=1, type=int)
    parser.add_argument('--sacae_autoencoder_lr', default=1e-5, type=float)
    parser.add_argument('--sacae_autoencoder_beta', default=0.9, type=float)
    parser.add_argument('--sacae_encoder_tau', default=0.05, type=float)
    parser.add_argument('--sacae_red_weight', default=None, type=float)

    # drq & atc
    parser.add_argument('--image_pad', default=4, type=int)

    # atc
    parser.add_argument('--atc_update_freq', default=1, type=int)
    parser.add_argument('--atc_lr', default=1e-3, type=float)
    parser.add_argument('--atc_beta', default=0.9, type=float)
    parser.add_argument('--atc_encoder_tau', default=0.01, type=float)
    parser.add_argument('--atc_target_update_freq', default=1, type=int)
    parser.add_argument('--atc_encoder_feature_dim', default=128, type=int)
    parser.add_argument('--atc_hidden_feature_dim', default=512, type=int)
    parser.add_argument('--atc_rl_clip_grad_norm', default=1000000, type=float)
    parser.add_argument('--atc_cpc_clip_grad_norm', default=10, type=float)

    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='./log', type=str)
    parser.add_argument('--save_tb', default=True, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=True, action='store_true')
    parser.add_argument('--save_model', default=True, action='store_true')
    parser.add_argument('--save_best_model', default=True, action='store_true')
    parser.add_argument('--detach_encoder', default=False, action='store_true')
    parser.add_argument('--log_interval', default=25, type=int)
    parser.add_argument('--tag', default='', type=str)
    parser.add_argument('--robot_shape', default=0, type=int)
    parser.add_argument('--robot_feature_dim', default=None, type=str)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--load_model', default=None, type=str)

    #preferenece reward
    parser.add_argument('--pr_files', default=None, type=str)
    parser.add_argument('--pr_size', default=1, type=int)
    parser.add_argument('--pr_env', default=None, type=str)
    parser.add_argument('--pr_alpha', default=1.0, type=float)
    parser.add_argument('--pr_as_cost', default=False, action='store_true')
    parser.add_argument('--pr_adapt_alpha', default='constant', type=str)
    parser.add_argument('--pr_remove_barrier', default=False, action='store_true')
    parser.add_argument('--pr_sb3_ensemble', default=False, action='store_true')

    #wandb
    parser.add_argument('--wandb_project', default=None, type=str)
    parser.add_argument('--wandb_name', default=None, type=str)

    if argumentString == None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argumentString)
    
    # verification
    assert (args.agent in ['curl', 'sacae', 'sac', 'rad', 'drq', 'atc', 'sac_state'])
    assert args.sacae_red_weight == None or args.sacae_red_weight <= 3
    assert args.sacae_red_weight == None or args.sacae_red_weight >= 0
    assert args.cost in ['no_cost', 'reward', 'critic_train', 'critic_eval']
    assert args.pr_adapt_alpha in ['constant', 'reward_based', 'steps_based']

    if args.agent in ['curl', 'rad']:
        args.env_image_size = 100
        args.agent_image_size = 84
    elif args.agent in ['sacae', 'sac', 'drq', 'atc']:
        args.env_image_size = 84
        args.agent_image_size = 84
    
    if args.agent not in ['drq', 'atc']:
        args.image_pad = None

    return args

class Arguments:
    def __init__(self, file) -> None:
        with open(file, 'r') as f:
            self.__dict__ =  json.load(f)