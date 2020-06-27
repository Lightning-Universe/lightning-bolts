.. role:: hidden
    :class: hidden-section

Bolts Loggers
=============
The loggers in this package are being considered to be added to the main PyTorch Lightning repository.
These loggers may be more unstable, in development, or not fully tested yet.

.. note:: This module is a work in progress

---------

allegro.ai TRAINS
^^^^^^^^^^^^^^^^^

`allegro.ai <https://github.com/allegroai/trains/>`_ is a third-party logger.
To use :class:`~pl_bolts.loggers.TrainsLogger` as your logger do the following.
First, install the package:

.. code-block:: bash

    pip install trains

Then configure the logger and pass it to the :class:`~pl_bolts.trainer.trainer.Trainer`:

.. testcode::

    from pl_bolts.loggers import TrainsLogger
    trains_logger = TrainsLogger(
        project_name='examples',
        task_name='pytorch lightning test',
    )
    trainer = Trainer(logger=trains_logger)

.. testoutput::
    :options: +ELLIPSIS, +NORMALIZE_WHITESPACE
    :hide:

    TRAINS Task: ...
    TRAINS results page: ...

.. testcode::

    class MyModule(LightningModule):
        def __init__(self):
            some_img = fake_image()
            self.logger.experiment.log_image('debug', 'generated_image_0', some_img, 0)

.. seealso::
    :class:`~pl_bolts.loggers.TrainsLogger` docs.

---------

Your Logger
-----------
Add your loggers here!
