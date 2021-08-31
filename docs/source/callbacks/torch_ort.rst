==================
Torch ORT Callback
==================

`Torch ORT <https://cloudblogs.microsoft.com/opensource/2021/07/13/accelerate-pytorch-training-with-torch-ort/>`__ converts your model into an optimized ONNX graph, speeding up training & inference when using NVIDIA or AMD GPUs. See installation instructions `here <https://github.com/pytorch/ort#install-in-a-local-python-environment>`__.

This is primarily useful for when training with a Transformer model. The ORT callback works when a single model is specified as `self.model` within the ``LightningModule`` as shown below.

.. note::

    Not all Transformer models are supported. See `this table <https://github.com/microsoft/onnxruntime-training-examples#examples>`__ for supported models + branches containing fixes for certain models.

.. code-block:: python

    from pytorch_lightning import LightningModule, Trainer
    from transformers import AutoModel

    from pl_bolts.callbacks import ORTCallback


    class MyTransformerModel(LightningModule):

        def __init__(self):
            super().__init__()
            self.model = AutoModel.from_pretrained('bert-base-cased')

        ...


    model = MyTransformerModel()
    trainer = Trainer(gpus=1, callbacks=ORTCallback())
    trainer.fit(model)


For even easier setup and integration, have a look at our Lightning Flash integration for :ref:`Text Classification <lightning_flash:text_classification_ort>`, :ref:`Translation <lightning_flash:translation_ort>` and :ref:`Summarization <lightning_flash:summarization_ort>`.
