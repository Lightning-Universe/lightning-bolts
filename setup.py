#!/usr/bin/env python

import os
import re
from importlib.util import module_from_spec, spec_from_file_location

from setuptools import find_packages, setup

_PATH_ROOT = os.path.realpath(os.path.dirname(__file__))
_PATH_REQUIRE = os.path.join(_PATH_ROOT, "requirements")
_FREEZE_REQUIREMENTS = bool(int(os.environ.get("FREEZE_REQUIREMENTS", 0)))


def _load_py_module(fname, pkg="pl_bolts"):
    spec = spec_from_file_location(os.path.join(pkg, fname), os.path.join(_PATH_ROOT, "src", pkg, fname))
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


def _augment_requirement(ln: str, comment_char: str = "#", unfreeze: bool = True) -> str:
    """Adjust the upper version contrains.

    >>> _augment_requirement("arrow<=1.2.2,>=1.2.0  # anything", unfreeze=False)
    'arrow<=1.2.2,>=1.2.0'
    >>> _augment_requirement("arrow<=1.2.2,>=1.2.0  # strict", unfreeze=False)
    'arrow<=1.2.2,>=1.2.0  # strict'
    >>> _augment_requirement("arrow<=1.2.2,>=1.2.0  # my name", unfreeze=True)
    'arrow>=1.2.0'
    >>> _augment_requirement("arrow>=1.2.0, <=1.2.2  # strict", unfreeze=True)
    'arrow>=1.2.0, <=1.2.2  # strict'
    >>> _augment_requirement("arrow", unfreeze=True)
    'arrow'

    """
    # filer all comments
    if comment_char in ln:
        comment = ln[ln.index(comment_char) :]
        ln = ln[: ln.index(comment_char)]
        is_strict = "strict" in comment
    else:
        is_strict = False
    req = ln.strip()
    # skip directly installed dependencies
    if not req or (unfreeze and any(c in req for c in ["http:", "https:", "@"])):
        return ""

    # remove version restrictions unless they are strict
    if unfreeze and "<" in req and not is_strict:
        req = re.sub(r",? *<=? *[\d\.\*]+,? *", "", req).strip()

    # adding strict back to the comment
    if is_strict:
        req += "  # strict"

    return req


def _load_requirements(path_dir: str, file_name: str, unfreeze: bool = not _FREEZE_REQUIREMENTS) -> list:
    """Loading requirements from a file.

    >>> path_req = os.path.join(_PATH_ROOT, "requirements")
    >>> _load_requirements(path_req, "docs.txt")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['sphinx>=4.0', ...]

    """
    with open(os.path.join(path_dir, file_name)) as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = [_augment_requirement(ln, unfreeze=unfreeze) for ln in lines]
    reqs = [str(req) for req in reqs if req and not req.startswith("-r")]
    if unfreeze:
        # filter empty lines and containing @ which means redirect to some git/http
        reqs = [req for req in reqs if not any(c in req for c in ["@", "http://", "https://"])]
    return reqs


def _load_readme_description(path_dir: str, homepage: str, ver: str) -> str:
    """Load readme as decribtion.

    >>> _load_readme_description(_PATH_ROOT, "", "")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    '<div align="center">...'

    """
    path_readme = os.path.join(path_dir, "README.md")
    with open(path_readme, encoding="utf-8") as fo:
        text = fo.read()

    # drop images from readme
    text = text.replace("![PT to PL](docs/source/_images/general/pl_quick_start_full_compressed.gif)", "")

    # https://github.com/PyTorchLightning/pytorch-lightning/raw/master/docs/source/_images/lightning_module/pt_to_png
    github_source_url = os.path.join(homepage, "raw", ver)
    # replace relative repository path to absolute link to the release
    #  do not replace all "docs" as in the readme we reger some other sources with particular path to docs
    text = text.replace("docs/source/_images/", f"{os.path.join(github_source_url, 'docs/source/_images/')}")

    # readthedocs badge
    text = text.replace("badge/?version=stable", f"badge/?version={ver}")
    text = text.replace("lightning-bolts.readthedocs.io/en/stable/", f"lightning-bolts.readthedocs.io/en/{ver}")
    # codecov badge
    text = text.replace("/branch/master/graph/badge.svg", f"/release/{ver}/graph/badge.svg")
    # replace github badges for release ones
    text = text.replace("badge.svg?branch=master&event=push", f"badge.svg?tag={ver}")

    skip_begin = r"<!-- following section will be skipped from PyPI description -->"
    skip_end = r"<!-- end skipping PyPI description -->"
    # todo: wrap content as commented description
    return re.sub(rf"{skip_begin}.+?{skip_end}", "<!--  -->", text, flags=re.IGNORECASE + re.DOTALL)

    # # https://github.com/Borda/pytorch-lightning/releases/download/1.1.0a6/codecov_badge.png
    # github_release_url = os.path.join(homepage, "releases", "download", ver)
    # # download badge and replace url with local file
    # text = _parse_for_badge(text, github_release_url)


def _prepare_extras():
    extras = {
        "loggers": _load_requirements(path_dir=_PATH_REQUIRE, file_name="loggers.txt"),
        "models": _load_requirements(path_dir=_PATH_REQUIRE, file_name="models.txt"),
        "test": _load_requirements(path_dir=_PATH_REQUIRE, file_name="test.txt"),
        "typing": _load_requirements(path_dir=_PATH_REQUIRE, file_name="typing.txt"),
    }
    extras["extra"] = extras["models"] + extras["loggers"]
    extras["dev"] = extras["extra"] + extras["test"]
    return extras


about = _load_py_module("__about__.py")
long_description = _load_readme_description(
    _PATH_ROOT,
    homepage=about.__homepage__,
    ver=about.__version__,
)

# https://packaging.python.org/discussions/install-requires-vs-requirements /
# keep the meta-data here for simplicity in reading this file... it's not obvious
# what happens and to non-engineers they won't know to look in init ...
# the goal of the project is simplicity for researchers, don't want to add too much
# engineer specific practices
setup(
    name="lightning-bolts",
    version=about.__version__,
    description=about.__docs__,
    author=about.__author__,
    author_email=about.__author_email__,
    url=about.__homepage__,
    download_url="https://github.com/PyTorchLightning/lightning-bolts",
    license=about.__license__,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    zip_safe=False,
    keywords=["deep learning", "pytorch", "AI"],
    python_requires=">=3.8",
    setup_requires=["wheel"],
    install_requires=_load_requirements(_PATH_REQUIRE, "base.txt"),
    extras_require=_prepare_extras(),
    project_urls={
        "Bug Tracker": "https://github.com/PyTorchLightning/lightning-bolts/issues",
        "Documentation": "https://lightning-bolts.rtfd.io/en/latest/",
        "Source Code": "https://github.com/PyTorchLightning/lightning-bolts",
    },
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        # Pick your license as you wish
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
