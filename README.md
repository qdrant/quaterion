<p align="center">
  <img height="100" src="docs/imgs/logo.svg" alt="Quaterion">
</p>

<p align="center">
    <b>Blazing fast framework for fine-tuning Similarity Learning models</b>
</p>

<p align=center>
    <a href="https://pypi.org/project/quaterion"><img src="https://img.shields.io/pypi/v/quaterion?label=pypi" alt="Version" /></a>
    <a href="https://github.com/qdrant/quaterion/actions/workflows/test.yml"><img src="https://github.com/qdrant/quaterion/actions/workflows/test.yml/badge.svg" alt="Tests status"></a>
    <a href="https://qdrant.to/discord"><img src="https://img.shields.io/badge/Discord-Qdrant-5865F2.svg?logo=discord" alt="Discord"></a>
    <a href="https://quaterion.qdrant.tech"><img src="https://img.shields.io/badge/Learn-Docs%20%26%20Tutorials-success" alt="Docs & Tutorials" /></a>
</p>

>  A dwarf on a giant's shoulders sees farther of the two 

Quaterion is a framework for fine-tuning similarity learning models.
The framework closes the "last mile" problem in training models for semantic search, recommendations, anomaly detection, extreme classification, matching engines, e.t.c.

It is designed to combine the performance of pre-trained models with specialization for the custom task while avoiding slow and costly training.


## Features

* 🌀 **Warp-speed fast**: With the built-in caching mechanism, Quaterion enables you to train thousands of epochs with huge batch sizes even on *laptop GPU*.

<p align="center">
  <img alt="Regular vs Cached Fine-Tuning" src="https://storage.googleapis.com/quaterion/docs/new-cmp-demo.gif">
</p>

* 🐈‍ **Small data compatible**: Pre-trained models with specially designed head layers allow you to benefit even from a dataset you can label *in one day*.


* 🏗️ **Customizable**: Quaterion allows you to re-define any part of the framework, making it flexible even for large-scale and sophisticated training pipelines.


* 🌌 **Scalable**: Quaterion is built on top of [PyTorch Lightning](https://github.com/Lightning-AI/lightning) and inherits all its scalability, cost-efficiency, and reliability perks. 

## Installation

TL;DR:

For training:
```bash
pip install quaterion
```

For inference service:
```bash
pip install quaterion-models
```

---

Quaterion framework consists of two packages - `quaterion` and [`quaterion-models`](https://github.com/qdrant/quaterion-models).

Since it is not always possible or convenient to represent a model in ONNX format (also, it **is supported**), the Quaterion keeps a very minimal collection of model classes, which might be required for model inference, in a [separate package](https://github.com/qdrant/quaterion-models).

It allows avoiding installing heavy training dependencies into inference infrastructure: `pip install quaterion-models`

At the same time, once you need to have a full arsenal of tools for training and debugging models, it is available in one package: `pip install quaterion`


## Docs 📓

* [Quick Start](https://quaterion.qdrant.tech/getting_started/quick_start.html) Guide
* Minimal working [examples](./examples)

For a more in-depth dive, check out our end-to-end tutorials:

- Fine-tuning NLP models - [Q&A systems](https://quaterion.qdrant.tech/tutorials/nlp_tutorial.html)
- Fine-tuning CV models - [Similar Cars Search](https://quaterion.qdrant.tech/tutorials/cars-tutorial.html)

Tutorials for advanced features of the framework:

- [Cache tutorial](https://quaterion.qdrant.tech/tutorials/cache_tutorial.html) - How to make training fast.
- [Head Layers: Skip Connection](https://quaterion.qdrant.tech/tutorials/head_layers_skip_connection.html) - How to avoid forgetting while fine-tuning
- [Embedding Confidence](https://quaterion.qdrant.tech/tutorials/embedding_confidence.html) - how do I know that the model is sure about the output vector?
- [Vector Collapse Prevention](https://quaterion.qdrant.tech/tutorials/triplet_loss_trick.html) - how to prevent vector space collapse in Triplet Loss


## Community

* Join our [Discord channel](https://qdrant.to/discord)
* Follow us on [Twitter](https://qdrant.to/twitter)
* Subscribe to our [Newsletters](https://qdrant.to/newsletter)
* Write us an email [info@qdrant.tech](mailto:info@qdrant.tech)

## License

Quaterion is licensed under the Apache License, Version 2.0. View a copy of the [License file](https://github.com/qdrant/quaterion/blob/master/LICENSE).
