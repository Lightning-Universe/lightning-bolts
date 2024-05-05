import functools
import os
from typing import Any, Callable

from lightning_utilities.core.imports import ModuleAvailableCache, RequirementCache


# ToDo: replace with utils wrapper after 0.10 is released
def requires(*module_path_version: str) -> Callable:
    """Wrapper for enforcing certain requirements for a particular class or function."""

    def decorator(func: Callable) -> Callable:
        reqs = [
            ModuleAvailableCache(mod_ver) if "." in mod_ver else RequirementCache(mod_ver)
            for mod_ver in module_path_version
        ]
        available = all(map(bool, reqs))
        if not available:

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                msg = os.linesep.join([repr(r) for r in reqs if not bool(r)])
                raise ModuleNotFoundError(f"Required dependencies not available: \n{msg}")

            return wrapper
        return func

    return decorator
