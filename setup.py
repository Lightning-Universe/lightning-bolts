#!/usr/bin/env python

import os

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

try:
    import builtins
except ImportError:
    import __builtin__ as builtins

try:
    import pytorch_lightning  # noqa: F401
except ImportError:
    try:
        import pip
    except ImportError:
        raise ImportError('Missing `pip` to install custom dependencies.')
    pip.main(['install', 'pytorch-lightning>=1.1.0'])

# https://packaging.python.org/guides/single-sourcing-package-version/
# http://blog.ionelmc.ro/2014/05/25/python-packaging/

_PATH_ROOT = os.path.dirname(__file__)
builtins.__LIGHTNING_BOLT_SETUP__: bool = True

import pl_bolts  # noqa: E402


def _load_requirements(path_dir=_PATH_ROOT, file_name='requirements.txt', comment_char='#'):
    from pytorch_lightning.setup_tools import _load_requirements as _lreq
    return _lreq(path_dir=path_dir, file_name=file_name, comment_char=comment_char)


def _load_long_description():
    from pytorch_lightning.setup_tools import _load_long_description as _lld
    return _lld(_PATH_ROOT)


def _prepare_extras():
    extras = {
        'loggers': _load_requirements(path_dir=os.path.join(_PATH_ROOT, 'requirements'), file_name='loggers.txt'),
        'models': _load_requirements(path_dir=os.path.join(_PATH_ROOT, 'requirements'), file_name='models.txt'),
        'test': _load_requirements(path_dir=os.path.join(_PATH_ROOT, 'requirements'), file_name='test.txt'),
    }
    extras['extra'] = extras['models'] + extras['loggers']
    extras['dev'] = extras['extra'] + extras['test']
    return extras


# https://packaging.python.org/discussions/install-requires-vs-requirements /
# keep the meta-data here for simplicity in reading this file... it's not obvious
# what happens and to non-engineers they won't know to look in init ...
# the goal of the project is simplicity for researchers, don't want to add too much
# engineer specific practices
setup(
    name='pytorch-lightning-bolts',
    version=pl_bolts.__version__,
    description=pl_bolts.__docs__,
    author=pl_bolts.__author__,
    author_email=pl_bolts.__author_email__,
    url=pl_bolts.__homepage__,
    download_url='https://github.com/PyTorchLightning/pytorch-lightning-bolts',
    license=pl_bolts.__license__,
    packages=find_packages(exclude=['tests', 'docs']),
    long_description=_load_long_description(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    zip_safe=False,
    keywords=['deep learning', 'pytorch', 'AI'],
    python_requires='>=3.6',
    setup_requires=['pytorch-lightning>=1.1.0'],
    install_requires=_load_requirements(),
    extras_require=_prepare_extras(),
    project_urls={
        "Bug Tracker": "https://github.com/PyTorchLightning/pytorch-lightning-bolts/issues",
        "Documentation": "https://pytorch-lightning-bolts.rtfd.io/en/latest/",
        "Source Code": "https://github.com/PyTorchLightning/pytorch-lightning-bolts",
    },
    classifiers=[
        'Environment :: Console',
        'Natural Language :: English',
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Information Analysis',
        # Pick your license as you wish
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
