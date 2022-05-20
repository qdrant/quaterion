# Quaterion

>  A dwarf on a giant's shoulders sees farther of the two 

Quaterion is a framework for fine-tuning similarity learning models.
The framework closes the "last mile" problem in training models for semantic search, recommendations, anomaly detection, extreme classification, matching engines, e.t.c.

It is designed to combine the performance of pre-trained models with specialization for the custom task while avoiding slow and costly training.


## Features

* 🌀 **Warp-speed fast**: With the built-in caching mechanism, Quaterion enables you to train thousands of epochs with huge batch sizes even on *laptop GPU*.

<p align="center">
  <img alt="Regular vs Cached Fine-Tuning" src="./docs/imgs/merged-demo.gif">
</p>

* 🐈‍ **Small data compatible**: Pre-trained models with specially designed head layers allow you to benefit even from a dataset you can label *in one day*.


* 🏗️ **Customizable**: Quaterion allows you to re-define any part of the framework, making it flexible even for large-scale and sophisticated training pipelines.

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

## Architecture

Quaterion is built on top of [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) - a framework for high-performance AI research.
It takes care of all the tasks involved in constructing a training loops for ML models:

- Epochs management -> [[tutorial](https://pytorch-lightning.readthedocs.io/en/latest/model/train_model_basic.html)]
- Logging -> [[tutorial](https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html?highlight=logging)]
- Early Stopping -> [[tutorial](https://pytorch-lightning.readthedocs.io/en/latest/common/early_stopping.html)]
- Checkpointing -> [[tutorial](https://pytorch-lightning.readthedocs.io/en/latest/common/checkpointing.html)]
- Distributed training -> [[tutorial](https://pytorch-lightning.readthedocs.io/en/latest/clouds/cluster.html)]
- [And many more](https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction.html)

In addition to PyTorch Lightning functionality, Quaterion provides a scaffold for defining:

- Fine-tunable similarity learning models
  - Encoders and Head Layers
- Datasets and Data Loaders for representing similarity information
- Loss functions for similarity learning
- Metrics for evaluating model performance

<!--

<details>
    <summary>Imports and definitions</summary>
    
```python
import torch
from torch import nn
import torchvision
from quaterion import TrainableModel
from quaterion.loss import SimilarityLoss, TripletLoss

from quaterion_models.encoders import Encoder
from quaterion_models.heads import EncoderHead, SkipConnectionHead

class MobilenetV3Encoder(Encoder):
    """Example of an Encoder for images, initialized from the pre-trained model
    """
    def __init__(self, embedding_size: int):
        super().__init__()
        # Download and initialize pre-trained model
        self.encoder = torchvision.models.mobilenet_v3_small(pretrained=True)
        # We remove last layer of the model, so that it will return raw embeddings
        self.encoder.classifier = nn.Identity()

        self._embedding_size = embedding_size

    @property
    def trainable(self) -> bool:
        return False  # We will only tune the head layer

    @property
    def embedding_size(self) -> int:
        return self._embedding_size  # Output size of this encoder

    def forward(self, images):
        return self.encoder.forward(images)

```
</details>

```python

class Model(TrainableModel):
    def __init__(self, embedding_size: int, lr: float):
        self._embedding_size = embedding_size
        self._lr = lr
        super().__init__()

    def configure_encoders(self) -> Encoder:
        # Define one or multiple encoders for the input data.
        # Each encoder could represent its own part of the data, 
        # or different aspects of the same object.
        return MobilenetV3Encoder(self._embedding_size)

    def configure_head(self, input_embedding_size) -> EncoderHead:
        # Forward concatenated encoder output into final trainable layer
        return SkipConnectionHead(input_embedding_size)

    def configure_loss(self) -> SimilarityLoss:
        # Define which loss function to use during the fine-tuning.
        return TripletLoss()

    def configure_optimizers(self):
        # And also which optimizer to use
        return torch.optim.Adam(self.model.parameters(), self._lr)
```

-->
