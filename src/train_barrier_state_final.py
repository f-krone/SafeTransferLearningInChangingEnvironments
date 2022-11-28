import sys
sys.path.append('sac_ae')
from sac_ae.utils.argument import parse_args
from sac_ae.train import train

def main():
    num_experiment = 3
    project_name = 'fetch-push-barrier-final'
    agent = 'sac_state'
    seeds = [211755, 368011, 771143]

    for i in range(num_experiment):
        for cost in ['reward', 'critic_eval', 'critic_train']:
            args = parse_args('')

            args.work_dir = f'../output/final/{project_name}'
            args.exp_name = f'{agent}_cost-{cost}_{i}'
            args.agent = agent 
            args.cost = cost
            args.env_name = 'CustomFetchPushCostSmallDense-v0'
            args.batch_size = 256 
            args.critic_lr = 1e-4 
            args.actor_lr = 1e-4
            args.encoder_feature_dim = 1024 
            args.hidden_dim = '1024,512,256' 
            args.init_temperature = 0.5
            args.init_steps = 10000
            args.num_train_steps = 3000000
            args.critic_target_update_freq = 1 
            args.actor_update_freq = 1 
            args.eval_freq = 25000 
            args.num_eval_episodes = 10
            args.seed = seeds[i]

            args.wandb_project = project_name
            args.wandb_name = args.exp_name

            train(args)

if __name__ == '__main__':
    main()