import wandb
import numpy as np

import sys
sys.path.append('sac_ae')
from sac_ae.utils.argument import parse_args
from sac_ae.train import train

def main():
    ensemble_size = 3
    project_name = 'fetch-push-cost-ensemble'
    seeds = np.random.randint(10000, size=ensemble_size)
    for i in range(ensemble_size):
        args = parse_args('')

        args.work_dir = f'../output/{project_name}'
        args.exp_name = f'SAC_ensemble_{i}'
        args.agent = 'sac_state' 
        args.env_name = 'CustomFetchPushCostDense-v0' 
        args.cost = 'reward' 
        args.batch_size = 256 
        args.critic_lr = 1e-4 
        args.actor_lr = 1e-4 
        args.encoder_feature_dim = 1024 
        args.hidden_dim = '1024,512,256' 
        args.init_temperature = 0.5 
        args.num_train_steps = 4000000 
        args.init_steps = 10000 
        args.critic_target_update_freq = 1 
        args.actor_update_freq = 1 
        args.eval_freq = 25000 
        args.num_eval_episodes = 5
        args.seed = int(seeds[i])

        args.wandb_project = project_name
        args.wandb_name = args.exp_name

        train(args)

if __name__ == '__main__':
    main()