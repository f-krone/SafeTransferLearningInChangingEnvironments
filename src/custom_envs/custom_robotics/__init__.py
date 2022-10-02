from gym.envs.registration import register
import gym

for reward_type in ["sparse", "dense"]:
    suffix = "Dense" if reward_type == "dense" else ""
    kwargs = {
        "reward_type": reward_type,
    }

    env_name = "CustomFetchPickAndPlace{}-v0".format(suffix)
    if env_name in gym.envs.registry.env_specs:
        del gym.envs.registry.env_specs[env_name]

    # register the environment so we can play with it
    gym.register(
        id=env_name,
        entry_point="custom_robotics.robotics:FetchPickAndPlaceEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )

for reward_type in ["sparse", "dense"]:
    suffix = "Dense" if reward_type == "dense" else ""
    kwargs = {
        "reward_type": reward_type,
    }

    env_name = "CustomFetchReach{}-v0".format(suffix)
    if env_name in gym.envs.registry.env_specs:
        del gym.envs.registry.env_specs[env_name]

    # register the environment so we can play with it
    gym.register(
        id=env_name,
        entry_point="custom_robotics.robotics:FetchReachEnv",
        kwargs=kwargs,
        max_episode_steps=50,
    )

for reward_type in ["sparse", "dense"]:
    suffix = "Dense" if reward_type == "dense" else ""
    kwargs = {
        "reward_type": reward_type,
    }

    env_name = "CustomFetchPush{}-v0".format(suffix)
    if env_name in gym.envs.registry.env_specs:
        del gym.envs.registry.env_specs[env_name]

    # register the environment so we can play with it
    gym.register(
        id=env_name,
        entry_point="custom_robotics.robotics:FetchPushEnv",
        kwargs=kwargs,
        max_episode_steps=100,
    )

for reward_type in ["sparse", "dense"]:
    suffix = "Dense" if reward_type == "dense" else ""
    kwargs = {
        "reward_type": reward_type,
        "has_barrier": True
    }

    env_name = "CustomFetchPushCost{}-v0".format(suffix)
    if env_name in gym.envs.registry.env_specs:
        del gym.envs.registry.env_specs[env_name]

    # register the environment so we can play with it
    gym.register(
        id=env_name,
        entry_point="custom_robotics.robotics:FetchPushEnv",
        kwargs=kwargs,
        max_episode_steps=150,
    )

for reward_type in ["sparse", "dense"]:
    suffix = "Dense" if reward_type == "dense" else ""
    kwargs = {
        "reward_type": reward_type,
        "has_barrier": True,
        "barrier_size": 0.07
    }

    env_name = "CustomFetchPushCostSmall{}-v0".format(suffix)
    if env_name in gym.envs.registry.env_specs:
        del gym.envs.registry.env_specs[env_name]

    # register the environment so we can play with it
    gym.register(
        id=env_name,
        entry_point="custom_robotics.robotics:FetchPushEnv",
        kwargs=kwargs,
        max_episode_steps=150,
    )

for reward_type in ["sparse", "dense"]:
    suffix = "Dense" if reward_type == "dense" else ""
    kwargs = {
        "reward_type": reward_type,
        "bird_eye_view": True
    }

    env_name = "CustomFetchPushBirdEye{}-v0".format(suffix)
    if env_name in gym.envs.registry.env_specs:
        del gym.envs.registry.env_specs[env_name]

    # register the environment so we can play with it
    gym.register(
        id=env_name,
        entry_point="custom_robotics.robotics:FetchPushEnv",
        kwargs=kwargs,
        max_episode_steps=100,
    )