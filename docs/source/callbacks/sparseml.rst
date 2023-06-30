=================
SparseML Callback
=================

`SparseML <https://docs.neuralmagic.com/sparseml/>`__ allows you to leverage sparsity to improve inference times substantially.

SparseML requires you to fine-tune your model with the ``SparseMLCallback`` + a SparseML Recipe. By training with the ``SparseMLCallback``, you can leverage the `DeepSparse <https://github.com/neuralmagic/deepsparse>`__ engine to exploit the introduced sparsity, resulting in large performance improvements.

.. warning::

    The SparseML callback requires the model to be ONNX exportable. This can be tricky when the model requires dynamic sequence lengths such as RNNs.

To use leverage SparseML & DeepSparse follow the below steps:

1. Choose your Sparse Recipe
----------------------------

To choose a recipe, have a look at `recipes <https://docs.neuralmagic.com/sparseml/source/recipes.html>`__ and `Sparse Zoo <https://docs.neuralmagic.com/sparsezoo/>`__.

It may be easier to infer a recipe via the UI dashboard using `Sparsify <https://github.com/neuralmagic/sparsify>`__ which allows you to tweak and configure a recipe.
This requires to import an ONNX model, which you can get from your ``LightningModule`` by doing ``model.to_onnx(output_path)``.

2. Train with SparseMLCallback
------------------------------

.. testcode::
    :skipif: not _SPARSEML_TORCH_SATISFIED

    from pytorch_lightning import LightningModule, Trainer
    from pl_bolts.callbacks import SparseMLCallback

    class MyModel(LightningModule):
        ...

    model = MyModel()

    trainer = Trainer(
        callbacks=SparseMLCallback(recipe_path='recipe.yaml')
    )

3. Export to ONNX!
------------------

Using the helper function, we handle any quantization/pruning internally and export the model into ONNX format.
Note this assumes either you have implemented the property ``example_input_array`` in the model or you must provide a sample batch as below.

.. testcode::
    :skipif: not _SPARSEML_TORCH_SATISFIED

    import torch

    model = MyModel()
    ...

    # export the onnx model, using the `model.example_input_array`
    SparseMLCallback.export_to_sparse_onnx(model, 'onnx_export/')

    # export the onnx model, providing a sample batch
    SparseMLCallback.export_to_sparse_onnx(model, 'onnx_export/', sample_batch=torch.randn(1, 128, 128, dtype=torch.float32))


Once your model has been exported, you can import this into either `Sparsify <https://github.com/neuralmagic/sparsify>`__ or `DeepSparse <https://github.com/neuralmagic/deepsparse>`__.
