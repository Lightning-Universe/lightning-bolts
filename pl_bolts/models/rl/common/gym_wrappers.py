"""
Set of wrapper functions for gym environments taken from
https://github.com/Shmuma/ptan/blob/master/ptan/common/wrappers.py
"""
import collections

import numpy as np
import torch

from pl_bolts.utils import _GYM_AVAILABLE, _OPENCV_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _GYM_AVAILABLE:
    import gym.spaces
    from gym import make as gym_make
    from gym import ObservationWrapper, Wrapper
else:  # pragma: no cover
    warn_missing_pkg('gym')
    Wrapper = object
    ObservationWrapper = object

if _OPENCV_AVAILABLE:
    import cv2
else:  # pragma: no cover
    warn_missing_pkg('cv2', pypi_name='opencv-python')


class ToTensor(Wrapper):
    """For environments where the user need to press FIRE for the game to start."""

    def __init__(self, env=None):
        if not _GYM_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError('You want to use `gym` which is not installed yet.')

        super(ToTensor, self).__init__(env)

    def step(self, action):
        """Take 1 step and cast to tensor"""
        state, reward, done, info = self.env.step(action)
        return torch.tensor(state), torch.tensor(reward), done, info

    def reset(self):
        """reset the env and cast to tensor"""
        return torch.tensor(self.env.reset())


class FireResetEnv(Wrapper):
    """For environments where the user need to press FIRE for the game to start."""

    def __init__(self, env=None):
        if not _GYM_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError('You want to use `gym` which is not installed yet.')

        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        """Take 1 step"""
        return self.env.step(action)

    def reset(self):
        """reset the env"""
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class MaxAndSkipEnv(Wrapper):
    """Return only every `skip`-th frame"""

    def __init__(self, env=None, skip=4):
        if not _GYM_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError('You want to use `gym` which is not installed yet.')

        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        """take 1 step"""
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(ObservationWrapper):
    """preprocessing images from env"""

    def __init__(self, env=None):
        if not _OPENCV_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError('This class uses OpenCV which it is not installed yet.')

        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        """preprocess the obs"""
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        """image preprocessing, formats to 84x84"""
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


class ImageToPyTorch(ObservationWrapper):
    """converts image to pytorch format"""

    def __init__(self, env):
        if not _OPENCV_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError('This class uses OpenCV which it is not installed yet.')

        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=new_shape, dtype=np.float32)

    @staticmethod
    def observation(observation):
        """convert observation"""
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(ObservationWrapper):
    """scales the pixels"""

    @staticmethod
    def observation(obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(ObservationWrapper):
    """"Wrapper for image stacking"""

    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        self.buffer = None
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            old_space.low.repeat(n_steps, axis=0),
            old_space.high.repeat(n_steps, axis=0),
            dtype=dtype,
        )

    def reset(self):
        """reset env"""
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        """convert observation"""
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class DataAugmentation(ObservationWrapper):
    """
    Carries out basic data augmentation on the env observations
    - ToTensor
    - GrayScale
    - RandomCrop
    """

    def __init__(self, env=None):
        if not _GYM_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError('You want to use `gym` which is not installed yet.')

        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        """preprocess the obs"""
        return ProcessFrame84.process(obs)


def make_environment(env_name):
    """Convert environment with wrappers"""
    env = gym_make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)
