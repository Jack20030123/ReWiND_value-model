import random

import numpy as np
import torch as th
from gym import Env
import gym
from gym import spaces
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3.common.monitor import Monitor
from metaworld.envs import (
    ALL_V2_ENVIRONMENTS_GOAL_HIDDEN,
    ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
)

from envs.wrappers import *
from reward_model.env_reward_model import EnvRewardModel
from reward_model.policy_observation_encoder import PolicyObservationEncoder

import torch
from torchvision import transforms

environment_to_instruction = {
    "assembly-v2": "assembly",
    "basketball-v2": "play basketball",
    "bin-picking-v2": "pick bin",
    "box-close-v2": "closing box",
    "button-press-topdown-v2": "Press the button from top",
    "button-press-topdown-wall-v2": "Press the button from top",
    "button-press-v2": "Press the button from side",
    "button-press-wall-v2": "Press the button from side",
    "coffee-button-v2": "Press the coffee button",
    "coffee-pull-v2": "Pull the coffee cup",
    "coffee-push-v2": "Push the coffee cup",
    "dial-turn-v2": "Turn the dial",
    "disassemble-v2": "disassemble",
    "door-close-v2": "Close the door",
    "door-lock-v2": "Turn door lock counter-clockwise",
    "door-open-v2": "Open the door",
    "door-unlock-v2": "Turn door lock clockwise",
    "hand-insert-v2": "Pick up the block and insert it into the hole",
    "drawer-close-v2": "Close the drawer",
    "drawer-open-v2": "open drawer",
    "faucet-open-v2": "Open the faucet",
    "faucet-close-v2": "Close the faucet",
    "hammer-v2": "hammer nail",
    "handle-press-side-v2": "Press the handle from side",
    "handle-press-v2": "Press the handle",
    "handle-pull-side-v2": "Pull the handle up from the side",
    "handle-pull-v2": "Pull the handle",
    "lever-pull-v2": "pull lever",
    "peg-insert-side-v2": "Insert the peg",
    "pick-place-wall-v2": "Pick up the block and placing it to the goal position",
    "pick-out-of-hole-v2": "pick bin",
    "reach-v2": "Reach the goal",
    "push-back-v2": "Push the block back to the goal",
    "push-v2": "Push the block to the goal",
    "pick-place-v2": "Pick up the block and placing it to the goal position",
    "plate-slide-v2": "Slide the plate into the gate",
    "plate-slide-side-v2": "Slide the plate into the gate from the side",
    "plate-slide-back-v2": "Slide the plate out of the gate",
    "plate-slide-back-side-v2": "Slide the plate out of the gate from the side",
    "peg-unplug-side-v2": "unplug peg",
    "soccer-v2": "Slide the ball into the gate",
    "stick-push-v2": "Push the stick",
    "stick-pull-v2": "Pull the stick",
    "push-wall-v2": "push bin",
    "reach-wall-v2": "Reach the goal",
    "shelf-place-v2": "place bin to shelf",
    "sweep-into-v2": "Sweep the block into the hole",
    "sweep-v2": "sweep block",
    "window-open-v2": "Open the window",
    "window-close-v2": "Close the window",
}

instruction_to_environment = {v: k for k, v in environment_to_instruction.items()}


# Define a base environment for MetaWorld
class MetaworldBase(Env):
    def __init__(
        self,
        env_id,
        seed=0,
        goal_observable=False,
        random_reset="train",
        max_episode_steps=128,
        use_proprio=False,
        terminate_on_success=False,
    ):
        """
        Parameters
        ----------
        env_id : int
            index of the environment
        seed : int
            random seed
        goal_observable : bool
            whether the goal is observable
        random_reset : bool
            whether to randomly reset the environment
        """
        super(MetaworldBase, self).__init__()

        self.all_env_types = (
            ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
            if goal_observable
            else ALL_V2_ENVIRONMENTS_GOAL_HIDDEN
        )
        # print(self.all_env_types, env_id)
        if goal_observable:
            env_id = env_id + "-goal-observable"
            self.base_env = self.all_env_types[env_id](seed=seed)
        else:
            env_id = env_id + "-goal-hidden"
            self.base_env = self.all_env_types[env_id](seed=seed)

        self.max_episode_steps = max_episode_steps

        self.base_env = TimeLimit(
            self.base_env, max_episode_steps=self.max_episode_steps
        )

        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        self.image_keys = ["image"]
        self.cropper = transforms.CenterCrop(224)
        self.image_reward_idx = 0

        self.observation_space = gym.spaces.Dict(
            {
                "proprio": gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
                "image": gym.spaces.Box(
                    low=0, high=255, shape=(480, 640, 3), dtype=np.uint8
                ),
            }
        )

        self.rank = seed
        self.env_id = env_id
        self.random_reset = random_reset

        self.use_proprio = use_proprio
        self.terminate_on_success = terminate_on_success

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes for learning)
        """
        state, reward, done, info = self.base_env.step(action)

        obs = self.get_obs(state)

        # if success, we add "is_success" to the info
        if "success" in info and info["success"]:
            info["is_success"] = True
            if self.terminate_on_success:
                done = True
        else:
            info["is_success"] = False
        return obs, reward, done, info

    def get_obs(self, state):
        """
        Get the current observation of the environment.

        Returns:
            observation (object): agent's observation of the current environment
        """
        # state = self.base_env._get_obs(self.base_env.prev_time_step)
        obs = {}
        if self.use_proprio:
            obs["proprio"] = state[:4]

        if self.image_keys:
            image = self.render(mode="rgb_array")
            # if args.center_crop:
            image = (
                self.cropper(torch.Tensor(image).permute(2, 0, 1))
                .permute(1, 2, 0)
                .numpy()
                .astype(np.uint8)
            )
            obs["image"] = image

        return obs

    def reset(self):
        """
        Resets the environment and optionally resets the underlying environment with a random seed.

        Returns:
            observation (object): the initial observation
        """
        if self.random_reset == "train":
            self.rank = random.randint(100, 400)
            self.base_env = self.all_env_types[self.env_id](seed=self.rank)
            self.base_env = TimeLimit(
                self.base_env, max_episode_steps=self.max_episode_steps
            )
        elif self.random_reset == "eval":
            self.rank = random.randint(400, 500)
            self.base_env = self.all_env_types[self.env_id](seed=self.rank)
            self.base_env = TimeLimit(
                self.base_env, max_episode_steps=self.max_episode_steps
            )
        elif self.random_reset == "demo":
            self.rank = random.randint(0, 100)
            self.base_env = self.all_env_types[self.env_id](seed=self.rank)
            self.base_env = TimeLimit(
                self.base_env, max_episode_steps=self.max_episode_steps
            )

        state = self.base_env.reset()

        import gc
        gc.collect()
        
        obs = self.get_obs(state)

        return obs

    def render(self, mode="rgb_array"):
        """
        Render the environment.

        Returns:
            observation (object): the current observation
        """
        return self.base_env.render(mode)


    def close(self):
        """
        Closes the environment. This is used to clean up resources and shutdown any child processes.

        Returns:
            None
        """
        return self.base_env.close()


class MetaworldImageEmbeddingWrapper(gym.Wrapper):
    def __init__(self, env, reward_model, policy_observation_encoder=None):
        super(MetaworldImageEmbeddingWrapper, self).__init__(env)
        self.reward_model = reward_model
        self.policy_observation_encoder = policy_observation_encoder  # Independent policy observation encoder

        # The observation space is a dict
        # Let us add image_feature to the observation space

        current_obs_space = self.env.observation_space
        assert isinstance(current_obs_space, spaces.Dict), (
            "Observation space must be a Dict."
        )

        image_keys = self.env.image_keys

        # Define the new observation space
        new_spaces = current_obs_space.spaces.copy()
        
        # Add image features for reward calculation (using reward model encoder)
        new_spaces["reward_image_feature_0"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(reward_model.img_output_dim,),
            dtype=np.float32,
        )
        
        # Add image features for policy (must use policy observation encoder)
        if policy_observation_encoder is not None:
            new_spaces["policy_image_feature_0"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(policy_observation_encoder.img_output_dim,),
                dtype=np.float32,
            )
        else:
            raise ValueError("policy_observation_encoder is required for generating policy input features")

        # Set the updated observation space
        self.observation_space = spaces.Dict(new_spaces)

    def __getstate__(self):
        """Custom method for pickling - exclude reward_model which might contain unpicklable objects"""
        state = self.__dict__.copy()
        # Remove the reward_model which might not be picklable
        if "reward_model" in state:
            del state["reward_model"]
        return state

    def __setstate__(self, state):
        """Custom method for unpickling"""
        self.__dict__.update(state)
        # Set reward_model to None - it will need to be set again after unpickling
        self.reward_model = None

    def _observation(self, observation):
        image = observation["image"]
        image = image[None, None, :, :, :]
        
        # Encode image for reward calculation (using reward model encoder)
        reward_image_feature = self.reward_model.encode_images(image).squeeze()
        observation["reward_image_feature_0"] = reward_image_feature
        
        # Encode image for policy (must use policy observation encoder)
        if self.policy_observation_encoder is not None:
            policy_image_feature = self.policy_observation_encoder.encode_images(image).squeeze()
            observation["policy_image_feature_0"] = policy_image_feature
        else:
            raise ValueError("policy_observation_encoder is required for generating policy input features")
        
        # For backward compatibility, keep original image_feature_0 (using reward model encoding)
        observation["image_feature_0"] = reward_image_feature

        return observation

    def reset(self):
        obs = self.env.reset()
        obs = self._observation(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._observation(obs)
        return obs, reward, done, info

    def seed(self, seed=None):
        pass


# Example usage of the base environment and wrappers
def create_wrapped_env(
    env_id,
    reward_model,
    pca_model=None,
    language_features=None,
    text_instruction=None,
    use_time=False,
    monitor=False,
    goal_observable=False,
    success_bonus=0.0,
    is_state_based=False,
    mode="train",
    use_proprio=False,
    dense_rewards_at_end=False,
    action_chunk_size=1,
    logger=None,
    terminate_on_success=False,
    use_progress_diff=False,
):
    """
    Creates a wrapped MetaWorld environment with the given options.

    Args:
        env_id: The MetaWorld environment ID.
        pca_model: The PCA model to use for dimensionality reduction (optional).
        language_features: The language features to use for the environment (optional).
        sparse_reward: Whether to use sparse rewards (default=True).
        use_simulator_reward: Whether to use the simulator reward (default=False).
        use_time: Whether to add time to the observation (default=True).
        monitor: Whether to monitor the environment returns, rewards, etc. (default=False).
        terminate_on_success: Whether to terminate the episode on success (default=False).
    Returns:
        A function that returns the wrapped environment when called.
    """

    def _init():
        policy_observation_encoder = PolicyObservationEncoder(
            device=reward_model.device,
            batch_size=64
        )
        
        if mode == "eval":
            base_env = MetaworldBase(
                env_id,
                goal_observable=goal_observable,
                random_reset="eval",
                use_proprio=use_proprio,
                terminate_on_success=terminate_on_success,
            )
        elif mode == "train":
            base_env = MetaworldBase(
                env_id,
                goal_observable=goal_observable,
                random_reset="train",
                use_proprio=use_proprio,
                terminate_on_success=terminate_on_success,
            )
        elif mode == "demo":
            base_env = MetaworldBase(
                env_id,
                goal_observable=goal_observable,
                random_reset="demo",
                use_proprio=use_proprio,
                terminate_on_success=terminate_on_success,
            )
        else:
            raise ValueError("Invalid mode")

        if pca_model is not None:
            base_env = PCAReducerWrapper(base_env, pca_model)

        if use_time:
            base_env = TimeWrapper(base_env)

        # breakpoint()
        # This replaces the metaworld state-based input with an image embedding too

        dense_eval = True if (mode == "eval" or mode == "demo") else False

        base_env = MetaworldImageEmbeddingWrapper(base_env, reward_model, policy_observation_encoder)

        base_env = LearnedRewardWrapper(
            base_env,
            reward_model,
            is_state_based=is_state_based,
            language_features=language_features,
            text_instruction=text_instruction,
            dense_eval=dense_eval,
            use_progress_diff=use_progress_diff,
        )

        # This adds the language features to the observation
        if language_features is not None:
            base_env = LanguageWrapper(base_env, language_features)

        # Environment keeps an aggregate reward at each step and outputs it only when the episode ends
        if dense_rewards_at_end:
            base_env = RewardAtEndWrapper(base_env)

        base_env = FlattenDictObservationWrapper(base_env, use_proprio=use_proprio)

        if action_chunk_size > 1:
            base_env = ActionChunkingWrapper(
                base_env, chunk_size=action_chunk_size, n_action_steps=action_chunk_size
            )

        # else:
        #     # Then we are an EnvRewardModel
        #     if reward_model.name == 'sparse':
        #         use_sparse = True
        #     elif reward_model.name == 'dense':
        #         use_sparse = False
        #     base_env = RewardWrapper(base_env, sparse=use_sparse, success_bonus=reward_model.success_bonus)

        # add gym normalize reward wrapper

        if monitor:
            base_env = Monitor(base_env)

        if logger is not None:
            base_env = LoggingWrapper(base_env, logger, prefix=mode)
        # base_env = gym.wrappers.NormalizeReward(base_env)

        return base_env

    return _init


if __name__ == "__main__":
    env = MetaworldBase("door-open-v2", goal_observable=True)
    env.reset()
    env.render()
