.. PyTorchLightning-Bolts documentation master file, created by
   sphinx-quickstart on Wed Mar 25 21:34:07 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyTorchLightning-Bolts documentation
====================================

.. include:: intro.rst

.. toctree::
   :maxdepth: 1
   :name: start
   :caption: Start here

   introduction_guide

.. toctree::
   :maxdepth: 2
   :name: api
   :caption: Python API

   api/pl_bolts.datamodules

.. toctree::
   :maxdepth: 2
   :name: models
   :caption: Models

   api/pl_bolts.models

.. toctree::
   :maxdepth: 2
   :name: callbacks
   :caption: Callbacks

   api/pl_bolts.callbacks

.. toctree::
   :maxdepth: 2
   :name: loggers
   :caption: Loggers

   api/pl_bolts.loggers


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. This is here to make sphinx aware of the modules but not throw an error/warning
.. toctree::
   :hidden:

   readme
   api/pl_bolts.models
   api/pl_bolts.callbacks
   api/pl_bolts.loggers
