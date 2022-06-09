# Contributing to Quaterion
We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features

## We Develop with GitHub
We use github to host code, to track issues and feature requests, as well as accept pull requests.

## We Use [Github Flow](https://guides.github.com/introduction/flow/index.html), So All Code Changes Happen Through Pull Requests
Pull requests are the best way to propose changes to the codebase (we use [Github Flow](https://guides.github.com/introduction/flow/index.html)).
We actively welcome your pull requests:

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. Ensure the test suite passes.
4. Make sure your code lints (ToDo).
5. Make sure that commits have a reference to related issue (e.g. `Fix model training #num_of_issue`)
6. Issue that pull request!

## Any contributions you make will be under the Apache License 2.0
In short, when you submit code changes, your submissions are understood to be under the same [Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using Github's [issues](https://github.com/qdrant/quaterion/issues)
We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/qdrant/quaterion/issues/new); it's that easy!

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Coding Style

1. We use [PEP8](https://www.python.org/dev/peps/pep-0008/) code style
2. We use [Python Type Annotations](https://docs.python.org/3/library/typing.html) whenever it is necessary
   1. If your IDE cannot infer type of some variable, it is a good sign to add some more type annotations
3. We document tensor transformations - type of tensors are usually not enough for comfortable understanding of the code
4. We prefer simplicity and practical approach over kaggle-level state-of-the-art accuracy
   1. If some modules or loss functions have complicated interface, dependencies, or just very complicated internally - we would prefer to keep them outside Quaterion.

## License
By contributing, you agree that your contributions will be licensed under its Apache License 2.0.
