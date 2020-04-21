#!/usr/bin/env python

import os
from io import open
# Always prefer setuptools over distutils
from setuptools import setup, find_namespace_packages

try:
    import builtins
except ImportError:
    import __builtin__ as builtins

# https://packaging.python.org/guides/single-sourcing-package-version/
# http://blog.ionelmc.ro/2014/05/25/python-packaging/

PATH_ROOT = os.path.dirname(__file__)
builtins.__LIGHTNING_BOLT_SETUP__ = True
builtins.__LIGHTNING_SETUP__ = True

from pytorch_lightning import bolts  # noqa: E402


# https://packaging.python.org/discussions/install-requires-vs-requirements /
# keep the meta-data here for simplicity in reading this file... it's not obvious
# what happens and to non-engineers they won't know to look in init ...
# the goal of the project is simplicity for researchers, don't want to add too much
# engineer specific practices
setup(
    name='pytorch-lightning-bolts',
    version=bolts.__version__,
    description=bolts.__docs__,
    author=bolts.__author__,
    author_email=bolts.__author_email__,
    url=bolts.__homepage__,
    download_url='https://github.com/PyTorchLightning/pytorch-lightning-bolts',
    license=bolts.__license__,
    packages=find_namespace_packages(exclude=['tests', 'docs']),

    long_description=pytorch_lightning.bolts.__long_doc__,
    long_description_content_type='text/markdown',
    include_package_data=True,
    zip_safe=False,

    keywords=['deep learning', 'pytorch', 'AI'],
    python_requires='>=3.6',
    setup_requires=['pytorch-lightning>=0.7.1'],
    install_requires=['pytorch-lightning>=0.7.1'],

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
        # 'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
