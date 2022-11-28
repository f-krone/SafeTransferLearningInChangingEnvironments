# Safe Transfer Learning For Robotic Environments

Deep reinforcement learning has shown impressive results when using complex features as observations. In recent years, the performance of agents using high dimensional image observations was improved significantly. However, it still lacks behind for complex tasks. In this work, we explore a novel approach to guide the exploration of an agent. We show that this can be utilized to significantly improve the performance of agents trained on high dimensional image observations. We call our approach the Preference Reward algorithm. Our approach uses reward shaping and can therefore be used with any reinforcement learning algorithm. The second major problem we focused on in this work is safe reinforcement learning. In robotic environments, the agent must be constrained to not reach certain states that for example could hurt humans or the robot itself. We propose two new algorithms to train agents with such constraints in mind. The algorithms are called Safety Training and Safety Evaluation. Our algorithms are designed around actor critic methods. Specifically, we evaluated them with a Soft Actor-Critic (SAC).

# Table of Contents
1. [Prerequisits](#prerequisits)
1. [Training](#training)
    1. [FetchReach](#fetchreach)
    1. [FetchPush](#fetchpush)
    1. [FetchPushBarrier - State](#fetchpushbarrier---state)
    1. [FetchPushBarrier - Pixel](#fetchpushbarrier---pixel)
1. [Credits](#credits)

# Prerequisits
1. Install mujoco from [https://github.com/deepmind/mujoco/releases](https://github.com/deepmind/mujoco/releases). This work was done with version 2.1.0. Newer versions might not work.
1. Install dependencies from requirements.txt
    ```shell
    pip install -r requirements.txt
    ```
1. Install pytorch with the correct cuda version
    ```shell
    pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
    ```
1. Install the custom environments
    ```shell
    cd src/custom_envs
    pip install -e .
    ```

# Training
In this work, we trained agents for four sets of experiments: FetchReach, FetchPush, FetchPushBarrier - State, and FetchPushBarrier - Pixel.
## FetchReach
To recreate the results on the FetchReach environment, run the following commands:
```shell
cd src/
python train_reach_final.py
```
Note that the environments need to be rendered for the pixel observations and depending on the machine, *xvfb* might be required to run the training.
The results are logged to a tensorboard in `output/final/fetch-reach-final/<run>/tb` as well as uploaded to Weights & Biases.

## FetchPush
The training on FetchPush takes several days for each setup. To spread the load on multiple amchines, the training was started without a script using the following commands:
```shell
cd src/sac_ae/
python train.py --agent drq --env_name CustomFetchPushDense-v0 --batch_size 256 --num_train_steps 2000000  --init_steps 1000 --critic_target_update_freq 1 --actor_update_freq 1 --discount=0.85 --encoder_feature_dim=1024 --hidden_dim=512,256,128 --init_temperature=0.5 --robot_shape 10 --robot_feature_dim 64,128 --eval_freq 25000 --num_eval_episodes 10 --wandb_project fetch-push-final --wandb_name <wandb_run_name> --exp_name <run_name> --work_dir ../../output/final/fetch-push-final --seed <seed>
```
Fill in the placeholders `<wandb_run_name>`, `<run_name>`, `<seed>`.<br>

To train with the preference reward algorithm, add the following parameters:
```shell
--pr_files ../../teachers/push/SAC_ensemble_ --pr_size 3 --pr_model_name model.pt
```
To train with an adaptive alpha additionally add these parameters:
```shell
--pr_adapt_alpha reward_based --pr_adapt_alpha_reward_min -20
```
Alternatively add this for a fixed alpha:
```shell
--pr_alpha <alpha>
```
Set the `<alpha>` to 0.5 or 1.0 to recreate the experiments.<br>
The results are logged to a tensorboard in `output/final/fetch-push-final/<run_name>/tb` as well as uploaded to Weights & Biases.

## FetchPushBarrier - State

To recreate the results on the FetchPushBarrier environment using state observations, run the following commands:
```shell
cd src/
python train_barrier_state_final.py
```
The results are logged to a tensorboard in `output/final/fetch-push-barrier-final/<run>/tb` as well as uploaded to Weights & Biases.

## FetchPushBarrier - Pixel

The training on FetchPushBarrier using pixel observations takes several days for each setup. To spread the load on multiple amchines, the training was started without a script using the following commands:
```shell
cd src/sac_ae/
python train.py --agent drq --env_name CustomFetchPushCostSmallDense-v0 --batch_size 256 --num_train_steps 2000000 --pr_files ../../teachers/push_barrier/SAC_ensemble_ --pr_model_name model.pt --pr_alpha <alpha> --init_steps 1000 --critic_target_update_freq 1 --actor_update_freq 1 --actor_lr=0.00001 --critic_lr=0.00001 --discount=0.85 --encoder_feature_dim 1024 --hidden_dim 1024,512,256 --init_temperature=0.5 --robot_shape 10 --robot_feature_dim 64,128 --cost <cost> --eval_freq 25000 --num_eval_episodes 10 --wandb_project fetch-push-barrier-drq-final --wandb_name <wandb_run_name> --exp_name <run_name> --work_dir ../../output/final/fetch-push-barrier-drq-final --seed <seed>
```
Fill in the placeholders `<wandb_run_name`, `<run_name>`, `<seed>`.<br>
Set the placeholders `<alpha>` to *0.5* or *1.0* and `<cost>` to *reward* or *critic_train*.<br>
The results are logged to a tensorboard in `output/final/fetch-push-barrier-drq-final/<run_name>/tb` as well as uploaded to Weights & Biases.

# Credits

- The environemnts in the `src/custom_envs` folder are modified versions from the OpenAI gym robotics environments. The original implementation can be found here: [OpenAI gym](https://github.com/openai/gym/tree/v0.21.0/gym/envs/robotics).
- The SAC implementation in `src/sac_ae` is a heavily modified version of the implementation found here: [RL Algorithms for Visual Continuous Control](https://github.com/KarlXing/RL-Visual-Continuous-Control)