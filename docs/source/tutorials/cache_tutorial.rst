The cache tutorial
++++++++++++++++++

Main idea
=========
One of the most intriguing features in Quaterion is the caching mechanism.
A tool which can make experimenting blazing fast.

During fine-tuning you use a pretrained model and attach one or more layers upon it.
The most resource intensive part of such a model is running the data through these pretrained layers.
They have a really huge number of parameters.

However, it is not always necessary to tune these pretrained parameters and calculate gradients for them.
You might be able to train faster and get similar quality if you freeze them.

In Quaterion, the underlying pre-trained models used for producing embeddings.
For this reason, we call them encoders.
If encoder's weights are frozen, then it is deterministic and emits the same embeddings for the same input every epoch.
It opens a room for improvement.
We can calculate these embeddings one time and cache them for fast access.

This is the main idea of the cache.

The main `limitation` of its usage have been already mentioned - your `encoders have to be frozen`.

How to use it?
==============

`TrainableModel <quaterion.train.trainable_model.TrainableModel>`_ has
`configure_caches <quaterion.train.trainable_model.html#quaterion.train.trainable_model.TrainableModel.configure_caches>`_
method which you need to override for cache usage.

This method should return `CacheConfig <quaterion.train.cache.cache_config.CacheConfig>`_ instance containing cache settings.

.. code-block:: python
    :caption: configure_caches definition

    def configure_caches(self) -> Optional[CacheConfig]:
        return CacheConfig(...)

Cache settings can be divided into parts according to their purpose:

1. Manage speed of initial cache filling
2. Choose storage for embeddings
3. Customize how an object is cached

Options overview
----------------

The first part of the options includes ``batch_size`` and ``num_workers`` - these are directly passed to dataloader and used in the process of filling the cache.

``batch_size`` does not affect training stage at all, you can tune it freely.

``num_workers`` can decrease time of cache filling, but increase training time.

.. code-block:: python
    :caption: tune cache filling speed

    def configure_caches(self) -> Optional[CacheConfig]:
        return CacheConfig(
                batch_size=16,
                num_workers=2
            )

Storage settings are ``cache_type``, ``mapping`` and ``save_dir``.

The first two options configure the device your cached embeddings will be stored during the run.

``cache_type`` set a default storage type for all encoders which will be cached.

``mapping`` provides a way to specify devices per encoder.

Currently, you can store your embeddings on `CPU` or on `GPU`.

The latter option, ``save_dir``, sets up a directory on disk to store embeddings on subsequent runs.
If you don't specify a directory for saving embeddings, they won't be saved to disk.

.. code-block:: python
    :caption: tune storage

    def configure_caches(self) -> Optional[CacheConfig]:
        return CacheConfig(
                cache_type=CacheType.GPU, # GPU as a default storage for frozen encoders embeddings
                mapping={"image_encoder": CacheType.CPU}  # Store `image_encoder` embeddings on CPU
                save_dir='cache'
            )

The third part of the cache settings is a kind of advanced one.

By default, `key_extractor` uses index of an item in a dataset as a key in the cache.
This is usually sufficient, however, it has its drawbacks that you may want to avoid.

For instance, in some cases data-independent keys may not be acceptable or desirable, and you may want to establish such a connection between them.

You can provide custom ``key_extractors`` and extract keys from features in your own way to obtain desired properties.
(Here, by features we mean raw data written into a corresponding ``obj`` field in ``SimilaritySample``)
Custom `key_extractor` also limits cache capabilities likewise ``num_workers``.
See the details in :ref:`Subsequent ideas`.

.. code-block:: python
    :caption: provide custom key extractor

    def configure_caches(self) -> Optional[CacheConfig]:
        def custom_key_extractor(feature):
            return feature['filename']

        return CacheConfig(
                key_extractor=custom_key_extractor  # use feature's filename as a key
            )

.. _Subsequent ideas:

Subsequent ideas
================

Despite eliminating the most time-consuming operations via cache, there may still be places that prevent your training loop from speeding up.

What does the data we extract from the dataset contain? - Labels and features.

In a typical setup, we use features only to create embeddings.
Assume we already read all the features and stored embeddings, it's time to train.

During training we need to retrieve labels from the dataset for each sample to form a batch.
This can include `I/O`, which is often the bottleneck.
Just imagine that you need to read an image every time you want to get labels.
Sounds wasteful.

A possible improvement here is to avoid reading the dataset and keep the labels during cache filling too.
It will be done automatically if your setup meets several conditions:

1. Cache is enabled.
2. All of encoders are frozen.
3. Multiprocessing is not used.
4. Key extraction is not overridden.

If all points are met, you will get a noticeable increase in speed.

Comprehensive example
=====================

Now that we know about all the options and limitations of the cache, we can take a look at a more comprehensive example.

.. code-block:: python
    :caption: comprehensive example

    def configure_caches(self) -> Optional[CacheConfig]:
        def custom_key_extractor(self, feature):
            # let's assume that features is a row and its first 10 symbols uniquely determines it
            return features[:10]

        return CacheConfig(
                mapping={
                    "content_encoder": CacheType.GPU,
                    # Store cache in GPU for `content_encoder`
                    "attitude_encoder": CacheType.CPU
                    # Store cache in RAM for `attitude_encoder`
                },
                batch_size=16,
                save_dir='cache_dir',  # directory on disk to store filled cache
                num_workers=2,  # Number of processes. Labels can't be cached if `num_workers` != 0
                key_extractors=custom_key_extractor  # Key extractor for each encoder.
                #  Equal to
                #  {
                #     "content_encoder": custom_key_extractor,
                #     "attitude_encoder": custom_key_extractor
                #  }
            )

In this setup we have 2 encoders: ``content_encoder`` and ``attitude_encoder``.
One of them stores its embeddings on the GPU, and the other on the CPU.

The cache is filled in batches of size 16.

After the cache is full, it will be stored in ``cache_dir`` under the current path.

The cache filling will be performed in two processes, and each encoder's embeddings will be stored under a key extracted using ``custom_key_extractor``.
The multiprocessing environment and the custom key extractor do not allow us to cache labels.
But with text data, it's not that important to avoid `I/O` because strings aren't as heavy as images and won't incur much overhead.

More examples can be found at
`configure_caches <quaterion.train.trainable_model.html#quaterion.train.trainable_model.TrainableModel.configure_caches>`_
documentation.

Full training pipeline utilising cache can be found in `NLP tutorial </tutorials/nlp_tutorial.html>`_.