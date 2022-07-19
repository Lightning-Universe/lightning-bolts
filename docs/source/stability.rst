.. _stability:

Bolts stability
===============

Currently we are going through major revision of Bolts to ensure all of the code is stable and compatible with the rest of the Lightning ecosystem.
For this reason, all of our features are either marked as stable or experimental. Stable features are implicit, experimental features are explicit.

At the beginning of the aforementioned revision, **ALL** of the features currently in the project have been marked as experimental and will undergo rigorous review and testing before they can be marked as stable.

This document is intended to help you know what to expect and to outline our commitment to stability.

Stable
______

For stable features, all of the following are true:

- the API isn’t expected to change
- if anything does change, incorrect usage will give a deprecation warning for **one major release** before the breaking change is made
- the API has been tested for compatibility with latest releases of PyTorch Lightning and Flash

Experimental
____________

For experimental features, any or all of the following may be true:

- the feature has unstable dependencies
- the API may change without notice in future versions
- the performance of the feature has not been verified
- the docs for this feature are under active development
