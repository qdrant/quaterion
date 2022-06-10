# Quick Start with Quaterion

There are a few concepts you need to know to get started with Quaterion:

## Similarity Samples and Data Loaders 

Unlike traditional classification or regression, 
similarity learning do not operate with specific target values.
Instead, it relies on the information about the similarity between objects.

Quaterion provides two primary methods of representing this "similarity" information.

### Similarity Pairs

[`SimilarityPairSample`](https://quaterion.qdrant.tech/quaterion.dataset.similarity_samples.html#quaterion.dataset.similarity_samples.SimilarityPairSample) - is a dataclass used to represent pairwise similarity between objects.

For example, if you want to train a food similarity model:

```python
data = [
    SimilarityPairSample(obj_a="cheesecake", obj_b="muffins", score=1.0),
    SimilarityPairSample(obj_a="cheesecake", obj_b="macaroons", score=1.0),
    SimilarityPairSample(obj_a="cheesecake", obj_b="candies", score=1.0),
    SimilarityPairSample(obj_a="lemon", obj_b="lime", score=1.0),
    SimilarityPairSample(obj_a="lemon", obj_b="orange", score=1.0),
]
```

Of course, you would also need to have negative examples - there are several strategies how to do it:

- Either specify negative samples explicitly:

```python
negative_data = [
    SimilarityPairSample(obj_a="cheesecake", obj_b="lemon", score=0.0),
    SimilarityPairSample(obj_a="orange", obj_b="macaroons", score=0.0),
    SimilarityPairSample(obj_a="lime", obj_b="candies", score=0.0)
]
```
- Or allow quaterion to assume, that all other samples pairs are negative, by using subgroups:

```python
data = [
    SimilarityPairSample(obj_a="cheesecake", obj_b="muffins", score=1.0, subgroup=10),
    SimilarityPairSample(obj_a="cheesecake", obj_b="macaroons", score=1.0, subgroup=10),
    SimilarityPairSample(obj_a="cheesecake", obj_b="candies", score=1.0, subgroup=10),
    SimilarityPairSample(obj_a="lemon", obj_b="lime", score=1.0, subgroup=11),
    SimilarityPairSample(obj_a="lemon", obj_b="orange", score=1.0, subgroup=11),
]
```

Quaterion will assume, that all samples with different subgroups are negative.

### Similarity Groups

Another handy way to provide similarity information is [`SimilarityGroupSample`](https://quaterion.qdrant.tech/quaterion.dataset.similarity_samples.html#quaterion.dataset.similarity_samples.SimilarityGroupSample).

It might be useful in following scenarios:

- Train similarity on multiple representations of the same object. E.g. multiple photos of same car.
- Convert labels into similarity samples - any classification dataset can be turned into a similarity dataset by assuming that objects of the same category are similar and of different categories - are not.

To use `SimilarityGroupSample` you need to assign the same `group_id` to objects belonging to the same group.

Example:

```python
data = [
    SimilarityGroupSample(obj="elon_musk_1.jpg", group=555),
    SimilarityGroupSample(obj="elon_musk_2.jpg", group=555),
    SimilarityGroupSample(obj="elon_musk_3.jpg", group=555),
    SimilarityGroupSample(obj="leonard_nimoy_1.jpg", group=209),
    SimilarityGroupSample(obj="leonard_nimoy_2.jpg", group=209),
]
```

### Data Loader

`SimilarityDataLoader` is a Data Loader that knows how to work correctly with SimilaritySamples.
There are `PairsSimilarityDataLoader` and `GroupSimilarityDataLoader` for `SimilarityPairSample` and `SimilarityGroupSample` respectively.

Wrap your dataset into one of the SimilarityDataLoader implementations to make it compatible with similarity learning:

```python
# Consumes data in format:
# {"description": "the thing I use for soup", "label": "spoon"} 
class JsonDataset(Dataset):
    def __init__(self, path: str):
        super().__init__()
        with open(path, "r") as f:
            self.data = [json.loads(line) for line in f.readlines()]

    def __getitem__(self, index: int) -> SimilarityGroupSample:
        item = self.data[index]
        return SimilarityGroupSample(obj=item, group=hash(item["label"]))

    def __len__(self) -> int:
        return len(self.data)

train_dataloader = GroupSimilarityDataLoader(JsonDataset('./my_data.json'), batch_size=128)
val_dataloader = GroupSimilarityDataLoader(JsonDataset('./my_data_val.json'), batch_size=128)
```

## Similarity Model and Encoders

[`SimilarityModel`](https://quaterion-models.qdrant.tech/quaterion_models.model.html#quaterion_models.model.SimilarityModel) - is
a model class, which manages all trainable layers.

The similarity model acts like an Encoder, which consists of other encoders, and a Head Layer, which combines outputs of encoder components.

```
 ┌─────────────────────────────────────┐
 │SimilarityModel                      │
 │ ┌─────────┐ ┌─────────┐ ┌─────────┐ │
 │ │Encoder 1│ │Encoder 2│ │Encoder 3│ │
 │ └────┬────┘ └────┬────┘ └────┬────┘ │
 │      │           │           │      │
 │      └────────┐  │  ┌────────┘      │
 │               │  │  │               │
 │           ┌───┴──┴──┴───┐           │
 │           │   concat    │           │
 │           └──────┬──────┘           │
 │                  │                  │
 │           ┌──────┴──────┐           │
 │           │    Head     │           │
 │           └─────────────┘           │
 └─────────────────────────────────────┘
```

Each encoder takes raw object data as an input and produces an embedding - a tensor of fixed length.

The rules for converting the raw input data into a tensor suitable for the neural network are defined separately in each encoder's [`collate_fn`](https://quaterion-models.qdrant.tech/quaterion_models.model.html#quaterion_models.model.SimilarityModel.get_collate_fn) function.

Let's define our simple encoder:

```python
class DescriptionEncoder(Encoder):
    def __init__(self, transformer: models.Transformer, pooling: models.Pooling):
        super().__init__()
        self.transformer = transformer
        self.pooling = pooling
        self.encoder = nn.Sequential(self.transformer, self.pooling)

    @property
    def trainable(self) -> bool:
        return False # Disable weights update for this encoder

    @property
    def embedding_size(self) -> int:
        return self.transformer.get_word_embedding_dimension()

    def forward(self, batch) -> Tensor:
        return self.encoder(batch)["sentence_embedding"]
    
    def collate_descriptions(self, batch: List[Any]) -> Tensor:
        descriptions = [record['description'] for record in batch]
        return self.transformer.tokenize(descriptions)
    
    def get_collate_fn(self) -> CollateFnType:
        return self.collate_descriptions

    def save(self, output_path: str):
        self.transformer.save(join(output_path, 'transformer'))
        self.pooling.save(join(output_path, 'pooling'))

    @classmethod
    def load(cls, input_path: str) -> Encoder:
        transformer = Transformer.load(join(input_path, 'transformer'))
        pooling = Pooling.load(join(input_path, 'pooling'))
        return cls(transformer=transformer, pooling=pooling)
```

Encoder is initialized with pre-trained layers `transformer` and `pooling`.
The initialization of the pre-trained components is defined outside the Encoder class. 
The encoder is designed to be used as a part of inference service, so it is important to keep training-related code outside. 

### Trainable Model

To properly initialize a model for training, Quaterion uses another entity - `TrainableModel`.
It contains methods that define the content of `SimilarityModel` as well as parameters for training.

```python
class Model(TrainableModel):
    def __init__(self, lr: float):
        self._lr = lr
        super().__init__()

    def configure_encoders(self) -> Union[Encoder, Dict[str, Encoder]]:
        pre_trained = SentenceTransformer("all-MiniLM-L6-v2")
        transformer, pooling = pre_trained[0], pre_trained[1]
        return DescriptionEncoder(transformer, pooling)

    def configure_head(self, input_embedding_size) -> EncoderHead:
        return SkipConnectionHead(input_embedding_size)

    def configure_loss(self) -> SimilarityLoss:
        return TripletLoss()

    def configure_optimizers(self):
        return torch.optim.Adam( self.model.parameters(), lr=self._lr)
```
`TrainableModel` is a descendant of [`pl.LightningModule`](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html) and serves the same function.

## Training

Now that we have the model and dataset, we can start training.
Training takes place at `Quaterion.fit`.

```python
model = Model(lr=0.01)

Quaterion.fit(
    trainable_model=model,
    trainer=None, # Use default trainer
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader
)
```

In the simplest case we can use the default trainer.
You will most likely need to change the training parameters, in which case we recommend overriding the default trainer parameters:

```python
trainer_kwargs = Quaterion.trainer_defaults()
trainer_kwargs['min_epochs'] = 10
trainer_kwargs['callbacks'].append(YourCustomCallback())
trainer = pl.Trainer(**trainer_kwargs)

Quaterion.fit(
    trainable_model=model,
    trainer=trainer, # Use custom trainer
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader
)
```

Read more about `pl.Trainer` at Pytorch Lightning [docs](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html)

After training is finished, we can save `SimilarityModel` for serving: 

```python
model.save_servable("./my_similarity_model")
```

## Further reading

Quick Start example is intended to give an idea of the structure of the framework and does not train any real model. 
It also does not cover important topics such as Caching, Evaluation, choosing loss functions and HeadLayers.

A working and more detailed example code can be found at:

* Minimal working [examples](./examples)

For a more in-depth dive, check out our end-to-end tutorials.

* Example: [fine-tuning NLP models](https://quaterion.qdrant.tech/tutorials/nlp_tutorial.html) - Q&A systems
* Example: [fine-tuning CV models](https://quaterion.qdrant.tech/tutorials/cars-tutorial.html) - similar cars search

Tutorials for advanced features of the framework:

* [Cache tutorial](https://quaterion.qdrant.tech/tutorials/cache_tutorial.html) - How to make training - warp-speed fast.