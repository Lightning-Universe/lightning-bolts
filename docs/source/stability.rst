.. _stability:

Bolts stability
===============

Currently we are going through major revision of Bolts to ensure all of the code is stable and compatible with the rest of the Lightning ecosystem.
For this reason, all of our features are either marked as stable or in need of review. Stable features are implicit, features to be reviewed are explicitly marked.

At the beginning of the aforementioned revision, **ALL** of the features currently in the project have been marked as to be reviewed and will undergo rigorous review and testing before they can be marked as stable. See `this GitHub issue <https://github.com/Lightning-AI/lightning-bolts/issues/819>`_ to check progress of the revision

This document is intended to help you know what to expect and to outline our commitment to stability.

Stable
______

For stable features, all of the following are true:

- the API isn’t expected to change
- if anything does change, incorrect usage will give a deprecation warning for **one minor release** before the breaking change is made
- the API has been tested for compatibility with latest releases of PyTorch Lightning and Flash

Under Review
____________

For features to be reviewed, any or all of the following may be true:

- the feature has unstable dependencies
- the API may change without notice in future versions
- the performance of the feature has not been verified
- the docs for this feature are under active development


Before a feature can be moved to Stable it needs to satisfy following conditions:

- Have appropriate tests, that will check not only correctness of the feature, but also compatibility with the current versions.
- Not have duplicate code accross Lightning ecosystem and more mature OSS projects.
- Pass a review process.
