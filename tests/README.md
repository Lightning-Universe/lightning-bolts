## How to run tests

The easiest way to run tests is by `make test`. Please note, that it may take quite some time, especially with unreliable internet connection since it is downloading datasets.

If you want to run specific tests, first make sure your environment is up to date by running `make env && make clean`. Then you can target specific tests such as `pytest "tests/models/gans/integration/test_gans.py::test_gan"`.

If you want to run doctests, run `make doctest` from root directory.

## Testing compatibility

Following PR [#844](https://github.com/Lightning-AI/lightning-bolts/pull/844), all of the new tests are required to use `catch_warnings` fixture in order to filter out compatibility issues. In the future, this fixture will be marked as `autouse=True`, however until then, please add them yourself.

### Example

```python
def test_this_thing(catch_warnings): ...
```
