# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
import inspect
from typing import Callable, Type, Union

from pytorch_lightning.utilities import rank_zero_warn


@functools.lru_cache()  # Trick to only warn once for each message
def _raise_experimental_warning(message: str, stacklevel: int = 6):
    rank_zero_warn(
        f"{message} The compatibility with other Lightning projects is not guaranteed and API may change at any time."
        "The API and functionality may change without warning in future releases. "
        "More details: https://lightning-bolts.readthedocs.io/en/latest/stability.html",
        stacklevel=stacklevel,
        category=UserWarning,
    )


def experimental(
    message: str = "This feature is currently marked as experimental.",
):
    """The experimental decorator is used to indicate that a particular feature is not properly reviewed and tested yet.
    A callable or type that has been marked as experimental will give a ``UserWarning`` when it is called or
    instantiated. This designation should be used following the description given in :ref:`stability`.
    Args:
        message: The message to include in the warning.
    Examples
    ________
    .. testsetup::
        >>> import pytest
    .. doctest::
        >>> from pl_bolts.utils.stability import experimental
        >>> @experimental()
        ... class MyExperimentalFeature:
        ...     pass
        ...
        >>> with pytest.warns(UserWarning, match="This feature is currently marked as experimental."):
        ...     MyExperimentalFeature()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        ...
        <...>
        >>> @experimental("This feature is currently marked as experimental with a message.")
        ... class MyExperimentalFeatureWithCustomMessage:
        ...     pass
        ...
        >>> with pytest.warns(UserWarning, match="This feature is currently marked as experimental with a message."):
        ...     MyExperimentalFeatureWithCustomMessage()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        ...
        <...>
    """

    def decorator(callable: Union[Callable, Type]):
        if inspect.isclass(callable):
            callable.__init__ = decorator(callable.__init__)
            return callable

        @functools.wraps(callable)
        def wrapper(*args, **kwargs):
            _raise_experimental_warning(message)
            return callable(*args, **kwargs)

        return wrapper

    return decorator
