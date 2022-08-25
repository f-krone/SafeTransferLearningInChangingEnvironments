from collections import deque
import numpy as np
import torch
import wandb
from utils.argument import parse_args
from utils.misc import set_seed_everywhere, make_dir, VideoRecorder, eval_mode
from utils.logger import Logger
from memory import ReplayBufferStorage, ReplayBufferStorageCost, make_replay_buffer
from model import make_model
from env import make_envs
from agent import make_agent
import time
import os
import json
from pathlib import Path

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

torch.backends.cudnn.benchmark = True

def evaluate(env, agent, video, num_episodes, L, step, log_cost, tag=None, low_cost_action=False, wandb_upload=False):
    episode_rewards = []
    episode_costs = []
    num_successes = 0
    video.init(enabled=True)
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_cost = 0
        info = {}
        while not done:
            with eval_mode(agent):
                if low_cost_action:
                    action = agent.select_low_cost_action(obs)
                else:
                    action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            video.record(env)
            episode_reward += reward
            if log_cost:
                episode_cost += info['cost']

        episode_rewards.append(episode_reward)
        episode_costs.append(episode_cost)
        if info.get('is_success'):
            num_successes += 1
    
    mean_reward = np.mean(episode_rewards)
    if L is not None:
        video.save(f'{step}.mp4')
        if wandb_upload:
            video.wandb_upload('eval/video')
        L.log(f'eval/success_rate', num_successes / num_episodes, step)
        L.log(f'eval/episode_reward', mean_reward, step)
        if log_cost:
            L.log('eval/episode_cost', np.mean(episode_costs), step)
    
    return mean_reward

def train(args, wandb_run=None):
    args.hidden_dim = list(map(lambda x: int(x), iter(args.hidden_dim.split(','))))
    if args.robot_feature_dim != None:
        args.robot_feature_dim = list(map(lambda x: int(x), iter(args.robot_feature_dim.split(','))))

    # prepare workspace
    set_seed_everywhere(args.seed)
    
    if args.exp_name == None:
        if args.agent == 'sac_state':
            ts = time.strftime("%m-%d-%H-%M", time.gmtime())
            exp_name = args.env_name + '-' + ts + 'state-b'  \
            + str(args.batch_size) + '-s' + str(args.seed)  + '-' + args.agent
        else:
            ts = time.strftime("%m-%d-%H-%M", time.gmtime())
            exp_name = args.env_name + '-' + ts + '-im' + str(args.env_image_size) +'-b'  \
            + str(args.batch_size) + '-s' + str(args.seed)  + '-' + args.agent
    else:
        exp_name = args.exp_name
    if args.tag:
        exp_name = exp_name + '-' + args.tag
    args.work_dir = args.work_dir + '/'  + exp_name
    make_dir(args.work_dir)
    video_dir = make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = make_dir(os.path.join(args.work_dir, 'model'))
    video = VideoRecorder(video_dir if args.save_video else None)
    
    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    
    run = None
    if (args.wandb_project != None and args.wandb_name != None):
        run = wandb.init(
            project=args.wandb_project,
            entity="f-krone",
            name=args.wandb_name,
            config=args,
            #sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
        wandb.tensorboard.patch(root_logdir=f'{args.work_dir}/tb', pytorch=True)
    else:
        print("Not using Weights&Biases. Please specify project and name.")
    L = Logger(args.work_dir, use_tb=args.save_tb, config=args.agent)

    # prepare env
    env = make_envs(args, False, use_state=args.agent == 'sac_state', logger=L)
    eval_env = make_envs(args, True, use_state=args.agent == 'sac_state')

    # prepare memory
    action_shape = env.action_space.shape
    if args.agent == 'sac_state':
        agent_obs_shape = env.observation_space.shape
        env_obs_shape = env.observation_space.shape
        args.agent_image_size = agent_obs_shape[0]
    elif args.cnn_3dconv:
        if args.robot_shape > 0:
            agent_obs_shape = env.observation_space['image'].shape
            env_obs_shape = env.observation_space['image'].shape
        else:
            agent_obs_shape = env.observation_space.shape
            env_obs_shape = env.observation_space.shape
    else:
        agent_obs_shape = (3*args.frame_stack, args.agent_image_size, args.agent_image_size)
        env_obs_shape = (3*args.frame_stack, args.env_image_size, args.env_image_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_cost = args.cost in ['critic_train', 'critic_eval']
    eval_low_cost_action = args.cost == 'critic_eval'
    log_cost = args.cost != 'no_cost'

    if save_cost:
        replay_storage = ReplayBufferStorageCost(Path(args.work_dir) / 'buffer', robot=args.robot_shape > 0, state=args.agent=="sac_state")
    else:
        replay_storage = ReplayBufferStorage(Path(args.work_dir) / 'buffer', robot=args.robot_shape > 0, state=args.agent=="sac_state")
    replay_buffer = None

    ep_success_buffer = deque(maxlen=100)
    
    model = make_model(agent_obs_shape, action_shape, args, device)

    # prepare agent
    agent = make_agent(
        model=model,
        device=device,
        action_shape=action_shape,
        args=args
    )

    if args.load_model != None:
        agent.load_model(args.load_model)

    if run != None or wandb_run != None:
        wandb.log({"model": str(model)})
    
    # run
    episode, episode_reward, episode_cost, done, info = 0, 0, 0, True, {}
    start_time = time.time()
    best_eval_reward = -1000

    for step in range(args.num_train_steps+1):
        # evaluate agent periodically

        if step > 0 and step % args.eval_freq == 0:
            L.log('eval/episode', episode, step)
            with torch.no_grad():
                eval_reward = evaluate(eval_env, agent, video, args.num_eval_episodes, L, step, log_cost,
                low_cost_action=eval_low_cost_action, wandb_upload=run != None or wandb_run != None)
            if args.save_model:
                agent.save_model(model_dir, step)
            if args.save_best_model and eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                agent.save_model(args.work_dir, 'best_model')
                print(f'Saving best model with reward: {eval_reward}')

        if done:
            if step > 0:
                # add the last observation for each episode
                if save_cost:
                    replay_storage.add(obs, None, None, None, True)
                else:
                    replay_storage.add(obs, None, None, True)
                ep_success_buffer.append(info.get('is_success'))
                if step % args.log_interval == 0:
                    L.log('train/episode_reward', episode_reward, step)
                    L.log('train/duration', time.time() - start_time, step)
                    L.log('train/success_rate', sum(1 for _ in filter(lambda x: x, iter(ep_success_buffer))) / len(ep_success_buffer), step)
                    if log_cost:
                        L.log('train/episode_cost', episode_cost, step)
                    L.dump(step)
                start_time = time.time()

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_cost = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            if replay_buffer is None:
                replay_buffer = make_replay_buffer(replay_dir=Path(args.work_dir) / 'buffer',
                                                   max_size=args.replay_buffer_capacity,
                                                   batch_size=args.batch_size,
                                                   num_workers=1,
                                                   save_snapshot=False,
                                                   nstep=1,
                                                   discount=args.discount,
                                                   obs_shape=env_obs_shape,
                                                   device=device,
                                                   image_size=args.agent_image_size,
                                                   image_pad=args.image_pad,
                                                   robot=args.robot_shape > 0,
                                                   save_cost=save_cost)


            num_updates = 1 if step > args.init_steps else args.init_steps
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        next_obs, reward, done, info = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
        episode_reward += reward
        if log_cost:
            episode_cost += info['cost']
        if save_cost:
            replay_storage.add(obs, action, reward, info['cost'], done_bool)
        else:
            replay_storage.add(obs, action, reward, done_bool)

        obs = next_obs
        episode_step += 1
    
    if run != None:
        run.finish()

def main():
    args = parse_args()
    train(args)
    
if __name__ == '__main__':
    main()