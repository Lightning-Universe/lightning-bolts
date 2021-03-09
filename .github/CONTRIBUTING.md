# Contributing

Welcome to the PyTorch Lightning community! We're building the most advanced research platform on the planet to implement the latest, best practices that the amazing PyTorch team rolls out!

## Bolts Design Principles

We encourage all sorts of contributions you're interested in adding! When coding for Bolts, please follow these principles.

#### Simple Internal Code

It's useful for users to look at the code and understand very quickly what's happening.
Many users won't be engineers. Thus we need to value clear, simple code over condensed ninja moves.
While that's super cool, this isn't the project for that :)

#### Force User Decisions To Best Practices

There are 1,000 ways to do something. However, eventually one popular solution becomes standard practice, and everyone follows.
We try to find the best way to solve a particular problem, and then force our users to use it for readability and simplicity.

When something becomes a best practice, we add it to the framework. This is usually something like bits of code in utils or in the model file that everyone keeps adding over and over again across projects. When this happens, bring that code inside the trainer and add a flag for it.

#### Backward-compatible API

We all hate updating our deep learning packages because we don't want to refactor a bunch of stuff. In bolts, we make sure every change we make which could break an API is backward compatible with good deprecation warnings.

#### Gain User Trust

As a researcher, you can't have any part of your code going wrong. So, make thorough tests to ensure that every implementation of a new trick or subtle change is correct.

#### Interoperability

PyTorch Lightning Bolts is highly interoperable with PyTorch Lightning and PyTorch.

---

## Contribution Types

We are always looking for help implementing new features or fixing bugs.

A lot of good work has already been done in project mechanics (requirements/base.txt, setup.py, pep8, badges, ci, etc...) so we're in a good state there thanks to all the early contributors (even pre-beta release)!

### Bug Fixes:

1. If you find a bug please submit a GitHub issue.

   - Make sure the title explains the issue.
   - Describe your setup, what you are trying to do, expected vs. actual behaviour. Please add configs and code samples.
   - Add details on how to reproduce the issue - a minimal test case is always best, colab is also great.
     Note, that the sample code shall be minimal and if needed with publicly available data.

2. Try to fix it or recommend a solution. We highly recommend to use test-driven approach:

   - Convert your minimal code example to a unit/integration test with assert on expected results.
   - Start by debugging the issue... You can run just this particular test in your IDE and draft a fix.
   - Verify that your test case fails on the master branch and only passes with the fix applied.

3. Submit a PR!

_**Note**, even if you do not find the solution, sending a PR with a test covering the issue is a valid contribution and we can help you or finish it with you :]_

### New Features:

1. Submit a GitHub issue - describe what is the motivation of such feature (adding the use case or an example is helpful).
2. Let's discuss to determine the feature scope.
3. Submit a PR! We recommend test driven approach to adding new features as well:

   - Write a test for the functionality you want to add.
   - Write the functional code until the test passes.

4. Add/update the relevant tests!

- [This PR](https://github.com/PyTorchLightning/pytorch-lightning/pull/2671) is a good example for adding a new metric, and [this one for a new logger](https://github.com/PyTorchLightning/pytorch-lightning/pull/2721).

### New Models:

PyTorch Lightning Bolts makes several research models for ready usage. Following are general guidelines for adding new models.

1. Models which are standard baselines
2. Whose results are reproduced properly either by us or by authors.
3. Top models which are not SOTA but highly cited for production usage / for other uses. (E.g. Mobile BERT, MobileNets, FBNets).
4. Do not reinvent the wheel, natively support torchvision, torchtext, torchaudio models.
5. Use open source licensed models.

Please raise an issue before adding a new model. Please let us know why the particular model is important for bolts. There are tons of models that keep coming. It is very difficult to support every model.

### Test cases:

Want to keep Lightning Bolts healthy? Love seeing those green tests? So do we! How to we keep it that way? We write tests! We value tests contribution even more than new features.

Tests are written using [pytest](https://docs.pytest.org/en/stable/). Tests in PyTorch Lightning bolts train model on a datamodule. Datamodule is lightning abstraction of representing dataloader and dataset. Model is checked by simply calling `.fit()` function over the datamodule.

Along with these we have tests for losses, callbacks and transforms as well.

Have a look at sample tests [here](https://github.com/PyTorchLightning/lightning-bolts/tree/master/tests).

After you have added the respective tests, you can run the tests locally with make script:

```bash
make test
```

Want to add a new test case and not sure how? [Talk to us!](https://join.slack.com/t/pytorch-lightning/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A)

**Note before submitting the PR, make sure you have run `make isort`.**
---

## Guidelines

For this section, we refer to read the [parent PL guidelines](https://pytorch-lightning.readthedocs.io/en/latest/CONTRIBUTING.html)

**Reminder**

All added or edited code shall be the own original work of the particular contributor.
If you use some third-party implementation, all such blocks/functions/modules shall be properly referred and if possible also agreed by code's author. For example - `This code is inspired from http://...`.
In case you adding new dependencies, make sure that they are compatible with the actual PyTorch Lightning license (ie. dependencies should be _at least_ as permissive as the PyTorch Lightning license).

### Question & Answer

1. **How can I help/contribute?**

   All help is extremely welcome - reporting bugs, fixing documentation, adding test cases, solving issues and preparing bug fixes. To solve some issues you can start with label [good first issue](https://github.com/PyTorchLightning/lightning-bolts/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) or chose something close to your domain with label [help wanted](https://github.com/PyTorchLightning/lightning-bolts/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22). Before you start to implement anything check that the issue description that it is clear and self-assign the task to you (if it is not possible, just comment that you take it and we assign it to you...).

2. **Is there a recommendation for branch names?**

   We do not rely on the name convention so far you are working with your own fork. Anyway it would be nice to follow this convention `<type>/<issue-id>_<short-name>` where the types are: `bugfix`, `feature`, `docs`, `tests`, ...

3. **I have a model in other framework than PyTorch, how do I add it here?**

   Since PyTorch Lightning is written on top of PyTorch. We need models in PyTorch only. Also, we would need same or equivalent results with PyTorch Lightning after converting the models from other frameworks.
