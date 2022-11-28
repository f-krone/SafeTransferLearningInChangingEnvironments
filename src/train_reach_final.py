import wandb
import numpy as np

import sys
sys.path.append('sac_ae')
from sac_ae.utils.argument import parse_args
from sac_ae.train import train

def main():
    num_experiment = 3
    project_name = 'fetch-reach-final'
    for agent in ['drq', 'sac', 'sacae']:
        for robot in [True, False]:
            for pr in [True, False]:
                for i in range(0, num_experiment):
                    pr = True
                    seed = np.random.randint(1000000)
                    args = parse_args('')

                    args.work_dir = f'../output/final/{project_name}'
                    args.exp_name = f'{agent}_robot-{robot}_pr-{pr}_a-05_{i}'
                    args.agent = agent 
                    args.env_name = 'CustomFetchReachDense-v0'
                    args.batch_size = 256 
                    args.critic_lr = 1e-4 
                    args.actor_lr = 1e-4 
                    args.discount = 0.85
                    args.encoder_feature_dim = 128
                    args.hidden_dim = '256,256'
                    args.init_temperature = 0.5 
                    args.num_train_steps = 100000
                    args.critic_target_update_freq = 1 
                    args.actor_update_freq = 1 
                    args.eval_freq = 25000
                    args.num_eval_episodes = 10
                    args.seed = seed

                    if pr:
                        args.pr_files = '../output/fetch-reach-ae-ensemble/SAC_ensemble_'
                        args.pr_model_name = 'best_model.pt'
                        args.pr_size = 3
                        args.pr_alpha = 0.5

                    if robot:
                        args.robot_shape = 10
                        args.robot_feature_dim = '32,64'

                    args.wandb_project = project_name
                    args.wandb_name = args.exp_name

                    if args.agent in ['sacae', 'sac', 'drq', 'atc']:
                        args.env_image_size = 84
                        args.agent_image_size = 84
                    if args.agent in ['drq', 'atc']:
                        args.image_pad = 4

                    train(args)

if __name__ == '__main__':
    main()