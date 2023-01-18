"""Set of wrapper functions for gym environments taken from
https://github.com/Shmuma/ptan/blob/master/ptan/common/wrappers.py."""
import collections

import numpy as np
import torch

from pl_bolts.utils import _GYM_AVAILABLE, _OPENCV_AVAILABLE
from pl_bolts.utils.stability import under_review
from pl_bolts.utils.warnings import warn_missing_pkg

if _GYM_AVAILABLE:
    import gymnasium as gym
    from gymnasium import ObservationWrapper, Wrapper
    from gymnasium import make as gym_make
else:  # pragma: no cover
    warn_missing_pkg("gym")
    Wrapper = object
    ObservationWrapper = object

if _OPENCV_AVAILABLE:
    import cv2
else:  # pragma: no cover
    warn_missing_pkg("cv2", pypi_name="opencv-python")


@under_review()
class ToTensor(Wrapper):
    """For environments where the user need to press FIRE for the game to start."""

    def __init__(self, env=None):
        if not _GYM_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `gym` which is not installed yet.")

        super().__init__(env)

    def step(self, action):
        """Take 1 step and cast to tensor."""
        obs, reward, done, truncated, info = self.env.step(action)
        return torch.tensor(obs), torch.tensor(reward), done, truncated, info

    def reset(self, **kwargs):
        """reset the env and cast to tensor."""
        obs, info = self.env.reset(**kwargs)
        return torch.tensor(obs), info


@under_review()
class FireResetEnv(Wrapper):
    """For environments where the user need to press FIRE for the game to start."""

    def __init__(self, env=None):
        if not _GYM_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `gym` which is not installed yet.")

        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        """Take 1 step."""
        return self.env.step(action)

    def reset(self, **kwargs):
        """reset the env."""
        obs, info = self.env.reset(**kwargs)
        obs, _reward, done, _truncated, _info = self.env.step(1)
        if done:
            obs, info = self.env.reset(**kwargs)
        obs, _reward, done, _truncated, _info = self.env.step(2)
        if done:
            obs, info = self.env.reset(**kwargs)
        return obs, info


@under_review()
class MaxAndSkipEnv(Wrapper):
    """Return only every `skip`-th frame."""

    def __init__(self, env=None, skip=4):
        if not _GYM_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `gym` which is not installed yet.")

        super().__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        """take 1 step."""
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, truncated, info

    def reset(self, **kwargs):
        """Clear past frame buffer and init.

        to first obs. from inner env.
        """
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info


@under_review()
class ProcessFrame84(ObservationWrapper):
    """preprocessing images from env."""

    def __init__(self, env=None):
        if not _OPENCV_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("This class uses OpenCV which it is not installed yet.")

        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        """preprocess the obs."""
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        """image preprocessing, formats to 84x84."""
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


@under_review()
class ImageToPyTorch(ObservationWrapper):
    """converts image to pytorch format."""

    def __init__(self, env):
        if not _OPENCV_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("This class uses OpenCV which it is not installed yet.")

        super().__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

    @staticmethod
    def observation(observation):
        """convert observation."""
        return np.moveaxis(observation, 2, 0)


@under_review()
class ScaledFloatFrame(ObservationWrapper):
    """scales the pixels."""

    @staticmethod
    def observation(obs):
        return np.array(obs).astype(np.float32) / 255.0


@under_review()
class BufferWrapper(ObservationWrapper):
    """Wrapper for image stacking."""

    def __init__(self, env, n_steps, dtype=np.float32):
        super().__init__(env)
        self.dtype = dtype
        self.buffer = None
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            old_space.low.repeat(n_steps, axis=0),
            old_space.high.repeat(n_steps, axis=0),
            dtype=dtype,
        )

    def reset(self, **kwargs):
        """reset env."""
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, observation):
        """convert observation."""
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


@under_review()
class DataAugmentation(ObservationWrapper):
    """Carries out basic data augmentation on the env observations.

    - ToTensor
    - GrayScale
    - RandomCrop
    """

    def __init__(self, env=None):
        if not _GYM_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `gym` which is not installed yet.")

        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        """preprocess the obs."""
        return ProcessFrame84.process(obs)


@under_review()
def make_environment(env_name):
    """Convert environment with wrappers."""
    if isinstance(env_name, str):
        env = gym_make(env_name)
    else:
        env = env_name

    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)
