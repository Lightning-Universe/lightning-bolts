#!/usr/bin/env python
<<<<<<< HEAD
=======
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
>>>>>>> 90c1c0f68b4983c685e9d009482890e578800439

import os

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

try:
    import builtins
except ImportError:
    import __builtin__ as builtins

# https://packaging.python.org/guides/single-sourcing-package-version/
# http://blog.ionelmc.ro/2014/05/25/python-packaging/
<<<<<<< HEAD

PATH_ROOT = os.path.dirname(__file__)
builtins.__LIGHTNING_BOLT_SETUP__ = True

import pl_bolts  # noqa: E402


def load_requirements(path_dir=PATH_ROOT, file_name='requirements.txt', comment_char='#'):
    with open(os.path.join(path_dir, file_name), 'r') as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        if comment_char in ln:  # filer all comments
            ln = ln[:ln.index(comment_char)].strip()
        if ln.startswith('http'):  # skip directly installed dependencies
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


def load_long_describtion():
    # https://github.com/PyTorchLightning/pytorch-lightning/raw/master/docs/source/_images/lightning_module/pt_to_pl.png
    url = os.path.join(pl_bolts.__homepage__, 'raw', pl_bolts.__version__, 'docs')
    text = open('README.md', encoding='utf-8').read()
    # replace relative repository path to absolute link to the release
    text = text.replace('](docs', f']({url}')
    # SVG images are not readable on PyPI, so replace them  with PNG
    text = text.replace('.svg', '.png')
    return text


extras = {
    'loggers': load_requirements(path_dir=os.path.join(PATH_ROOT, 'requirements'), file_name='loggers.txt'),
    'models': load_requirements(path_dir=os.path.join(PATH_ROOT, 'requirements'), file_name='models.txt'),
    'test': load_requirements(path_dir=os.path.join(PATH_ROOT, 'requirements'), file_name='test.txt'),
}
extras['extra'] = extras['models'] + extras['loggers']
extras['dev'] = extras['extra'] + extras['test']

=======
PATH_ROOT = os.path.dirname(__file__)
builtins.__LIGHTNING_SETUP__ = True

import pytorch_lightning  # noqa: E402
from pytorch_lightning.setup_tools import _load_long_description, _load_requirements  # noqa: E402

# https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras
# Define package extras. These are only installed if you specify them.
# From remote, use like `pip install pytorch-lightning[dev, docs]`
# From local copy of repo, use like `pip install ".[dev, docs]"`
extras = {
    # 'docs': load_requirements(file_name='docs.txt'),
    'examples': _load_requirements(path_dir=os.path.join(PATH_ROOT, 'requirements'), file_name='examples.txt'),
    'loggers': _load_requirements(path_dir=os.path.join(PATH_ROOT, 'requirements'), file_name='loggers.txt'),
    'extra': _load_requirements(path_dir=os.path.join(PATH_ROOT, 'requirements'), file_name='extra.txt'),
    'test': _load_requirements(path_dir=os.path.join(PATH_ROOT, 'requirements'), file_name='test.txt')
}
extras['dev'] = extras['extra'] + extras['loggers'] + extras['test']
extras['all'] = extras['dev'] + extras['examples']  # + extras['docs']

# These packages shall be installed only on GPU machines
PACKAGES_GPU_ONLY = (
    'horovod',
)
# create a version for CPU machines
for ex in ('cpu', 'cpu-extra'):
    kw = ex.split('-')[1] if '-' in ex else 'all'
    # filter cpu only packages
    extras[ex] = [pkg for pkg in extras[kw] if not any(pgpu.lower() in pkg.lower() for pgpu in PACKAGES_GPU_ONLY)]
>>>>>>> 90c1c0f68b4983c685e9d009482890e578800439

# https://packaging.python.org/discussions/install-requires-vs-requirements /
# keep the meta-data here for simplicity in reading this file... it's not obvious
# what happens and to non-engineers they won't know to look in init ...
# the goal of the project is simplicity for researchers, don't want to add too much
# engineer specific practices
setup(
<<<<<<< HEAD
    name='pytorch-lightning-bolts',
    version=pl_bolts.__version__,
    description=pl_bolts.__docs__,
    author=pl_bolts.__author__,
    author_email=pl_bolts.__author_email__,
    url=pl_bolts.__homepage__,
    download_url='https://github.com/PyTorchLightning/pytorch-lightning-bolts',
    license=pl_bolts.__license__,
    packages=find_packages(exclude=['tests', 'docs']),

    long_description=load_long_describtion(),
=======
    name="pytorch-lightning",
    version=pytorch_lightning.__version__,
    description=pytorch_lightning.__docs__,
    author=pytorch_lightning.__author__,
    author_email=pytorch_lightning.__author_email__,
    url=pytorch_lightning.__homepage__,
    download_url='https://github.com/PyTorchLightning/pytorch-lightning',
    license=pytorch_lightning.__license__,
    packages=find_packages(exclude=['tests', 'tests/*', 'benchmarks']),

    long_description=_load_long_description(PATH_ROOT),
>>>>>>> 90c1c0f68b4983c685e9d009482890e578800439
    long_description_content_type='text/markdown',
    include_package_data=True,
    zip_safe=False,

    keywords=['deep learning', 'pytorch', 'AI'],
    python_requires='>=3.6',
    setup_requires=[],
<<<<<<< HEAD
    install_requires=load_requirements(),
    extras_require=extras,

    project_urls={
        "Bug Tracker": "https://github.com/PyTorchLightning/pytorch-lightning-bolts/issues",
        "Documentation": "https://pytorch-lightning-bolts.rtfd.io/en/latest/",
        "Source Code": "https://github.com/PyTorchLightning/pytorch-lightning-bolts",
=======
    install_requires=_load_requirements(PATH_ROOT),
    extras_require=extras,

    project_urls={
        "Bug Tracker": "https://github.com/PyTorchLightning/pytorch-lightning/issues",
        "Documentation": "https://pytorch-lightning.rtfd.io/en/latest/",
        "Source Code": "https://github.com/PyTorchLightning/pytorch-lightning",
>>>>>>> 90c1c0f68b4983c685e9d009482890e578800439
    },

    classifiers=[
        'Environment :: Console',
        'Natural Language :: English',
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
<<<<<<< HEAD
        'Development Status :: 3 - Alpha',
=======
        'Development Status :: 4 - Beta',
>>>>>>> 90c1c0f68b4983c685e9d009482890e578800439
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
