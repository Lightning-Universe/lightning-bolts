from typing import Union, List

import numpy as np

class RunningStatistics:
    """
    Keeps track of first and second moments (mean and variance) of a streaming time series.
    Taken from https://github.com/joschu/modular_rl
    Math in http://www.johndcook.com/blog/standard_deviation/
    """

    def __init__(self, shape: int) -> None:
        self.n = 0
        self.mean = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x: Union[List, np.array]) -> None:
        x = np.asarray(x)
        assert x.shape == self.mean.shape
        self.n += 1
        if self.n == 1:
            self.mean[...] = x
        else:
            oldM = self.mean.copy()
            self.mean[...] = oldM + (x - oldM) / self.n
            self._S[...] = self._S + (x - oldM) * (x - self.mean)

    @property
    def var(self) -> np.array:
        return self._S / (self.n - 1) if self.n > 1 else np.square(self.mean)

    @property
    def std(self) -> np.array:
        return np.sqrt(self.var)


class ZFilter:
    """
    Normalizes variables using the mean and standard deviation by calculating them from running statistics.
    """

    def __init__(self, shape: int) -> None:
        self.running_statistics = RunningStatistics(shape)

    def __call__(self, x: Union[List, np.array]) -> Union[List, np.array]:
        self.running_statistics.push(x)
        x = x - self.running_statistics.mean
        x = x / (self.running_statistics.std + 1e-8)
        return x
