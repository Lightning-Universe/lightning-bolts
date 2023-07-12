import functools
import os
from typing import Callable, Any

from lightning_utilities.core.imports import ModuleAvailableCache, RequirementCache


def requires(*module_path_version: str) -> Callable:
    """Wrapper for enforcing certain requirements for a particular class or function."""

    def decorator(func: Callable) -> Callable:
        reqs = []
        for mod_ver in module_path_version:
            if "." in mod_ver:
                reqs.append(ModuleAvailableCache(mod_ver))
            else:
                reqs.append(RequirementCache(mod_ver))

        available = all(map(bool, reqs))
        if not available:

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                msg = os.linesep.join([repr(r) for r in reqs if not bool(r)])
                raise ModuleNotFoundError(f"Required dependencies not available: \n{msg}")

            return wrapper
        return func

    return decorator
