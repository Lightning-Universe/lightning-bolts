import os
PACKAGE_ROOT = os.path.dirname(__file__)

try:
    # This variable is injected in the __builtins__ by the build
    # process. It used to enable importing subpackages of skimage when
    # the binaries are not built
    __LIGHTNING_BOLT_SETUP__
except NameError:
    __LIGHTNING_BOLT_SETUP__ = False

if __LIGHTNING_BOLT_SETUP__:
    import sys  # pragma: no-cover
    sys.stdout.write(f'Partial import of `{__name__}` during the build process.\n')  # pragma: no-cover
    # We are not importing the rest of the lightning during the build process, as it may not be compiled yet
else:
    from pytorch_lightning.bolts.models.mnist_template import LitMNISTModel

    __all__ = [
        'LitMNISTModel'
    ]
