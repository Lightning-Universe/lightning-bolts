AsynchronousLoader
------------------
This dataloader behaves identically to the standard pytorch dataloader, but will transfer
data asynchronously to the GPU with training. You can also use it to wrap an existing dataloader.

.. note::

    We rely on the community to keep these updated and working. If something doesn't work, we'd really appreciate a contribution to fix!

Example:

.. code-block:: python

    dataloader = AsynchronousLoader(DataLoader(ds, batch_size=16), device=device)

    for b in dataloader:
        ...

.. autoclass:: pl_bolts.datamodules.async_dataloader.AsynchronousLoader
   :noindex:
