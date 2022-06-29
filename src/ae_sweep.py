import wandb

import sys
sys.path.append('sac_ae')
from sac_ae.utils.argument import parse_args
from sac_ae.train import train

def main():
    run = wandb.init(monitor_gym=True)
    project_name = run.project_name()
    wandb.tensorboard.patch(root_logdir=f'../output/sweeps/{project_name}/{run.id}/tb', pytorch=True)
    config = wandb.config
    args = parse_args()

    args.work_dir = f'../output/sweeps/{project_name}'
    args.exp_name = run.id

    args.save_video = False
    args.save_model = False

    args.env_name = 'CustomFetchReachDense-v0'
    args.batch_size = 256
    args.num_train_steps = 250000
    args.pr_files = '../output/fetch-reach-ensemble/SAC_ensemble_'
    args.pr_size = 3
    args.robot_shape = 10

    args.critic_lr = config.critic_lr
    args.actor_lr = config.actor_lr
    args.sacae_autoencoder_lr = config.sacae_autoencoder_lr
    args.sacae_update_freq = config.sacae_update_freq
    args.sacae_red_weight = config.sacae_red_weight
    args.hidden_dim = config.hidden_dim
    args.encoder_feature_dim = config.encoder_feature_dim
    args.discount = config.discount
    args.pr_alpha = config.pr_alpha

    train(args)

if __name__ == '__main__':
    main()