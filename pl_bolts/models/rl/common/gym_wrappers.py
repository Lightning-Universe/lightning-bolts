"""
Set of wrapper functions for gym environments taken from
https://github.com/Shmuma/ptan/blob/master/ptan/common/wrappers.py
and
    https://raw.githubusercontent.com/openai/baselines/7c520852d9cf4eaaad326a3d548efc915dc60c10/baselines/common/atari_wrappers.py

"""
import collections

import numpy as np
import torch

from pl_bolts.utils import _GYM_AVAILABLE, _OPENCV_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _GYM_AVAILABLE:
    import gym.spaces
    from gym import make as gym_make
    from gym import ObservationWrapper, RewardWrapper, Wrapper
else:  # pragma: no cover
    warn_missing_pkg('gym')
    Wrapper = object
    ObservationWrapper = object

if _OPENCV_AVAILABLE:
    import cv2
else:  # pragma: no cover
    warn_missing_pkg('cv2', pypi_name='opencv-python')


class NoopResetEnv(Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


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


class EpisodicLifeEnv(Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        if not _GYM_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError('You want to use `gym` which is not installed yet.')
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
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


class ClipRewardEnv(RewardWrapper):
    def __init__(self, env):
        super(ClipRewardEnv, self).__init__(env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class ProcessFrame84(ObservationWrapper):
    """preprocessing images from env"""

    def __init__(self, env=None):
        if not _OPENCV_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError('This class uses OpenCV which it is not installed yet.')

        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

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


class FrameStack(Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        super(FrameStack, self).__init__(env)
        self.k = k
        self.frames = collections.deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)),
                                                dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


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


class LazyFrames:
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


def make_atari_env(env_name):
    """Convert environment with wrappers"""
    # Modified from https://github.com/facebookresearch/torchbeast/blob/master/torchbeast/atari_wrappers.py#L289-L297
    env = gym_make(env_name)
    assert "NoFrameskip" in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env)
    return env


def make_deepmind_env(env_name, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    env = make_atari_env(env_name)
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env_name.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return ImageToPyTorch(env)


def make_environment(env_name):
    """Convert environment with wrappers"""
    # make_atari_environment is ready for Atari Game only.
    env = gym_make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)
