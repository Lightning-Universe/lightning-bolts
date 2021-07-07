import os
import warnings
from typing import Callable, Dict, Optional

MISSING_PACKAGE_WARNINGS: Dict[str, int] = {}

WARN_MISSING_PACKAGE = int(os.environ.get('WARN_MISSING_PACKAGE', False))


def warn_missing_pkg(
    pkg_name: str,
    pypi_name: Optional[str] = None,
    extra_text: Optional[str] = None,
    stdout_func: Callable = warnings.warn,
) -> int:
    """
    Template for warning on missing packages, show them just once.

    Args:
        pkg_name: Name of missing package
        pypi_name: In case that package name differ from PyPI name
        extra_text: Additional text after the base warning
        stdout_func: Define used function for streaming warning, use ``warnings.warn`` or ``logging.warning``

    Returns:
        Number of warning calls
    """
    if not WARN_MISSING_PACKAGE:
        return -1

    if pkg_name not in MISSING_PACKAGE_WARNINGS:
        extra_text = os.linesep + extra_text if extra_text else ''
        if not pypi_name:
            pypi_name = pkg_name
        stdout_func(
            f'You want to use `{pkg_name}` which is not installed yet,'
            f' install it with `pip install {pypi_name}`.' + extra_text
        )
        MISSING_PACKAGE_WARNINGS[pkg_name] = 1
    else:
        MISSING_PACKAGE_WARNINGS[pkg_name] += 1

    return MISSING_PACKAGE_WARNINGS[pkg_name]
