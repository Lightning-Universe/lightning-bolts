## Testing compatibility

Following PR [#844](https://github.com/Lightning-AI/lightning-bolts/pull/844), all of the new tests are required to use `catch_warnings` fixture in order to filter out compatibility issues. In the future, this fixture will be marked as `autouse=True`, however until then, please add them yourself.

### Example

```python
def test_this_thing(catch_warnings):
    ...
```
