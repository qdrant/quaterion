The cache tutorial
++++++++++++++++++

.. _Main idea:

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


.. image:: ../../imgs/merged-demo.gif
    :height: 250px
    :alt: regular vs cache gif


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

``num_workers`` determines number of processes to be used during cache filling.

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

The third part of the cache settings is a kind of advanced one and covered in :ref:`Limitations`.

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
It will be done automatically and bring a noticeable increase in training speed if cache is enabled and limitations described in the following chapter are met.


.. _Limitations:

Limitations
===========

As it was mentioned in :ref:`Main idea`, the main limitation of using cache is that encoders which is meant to be cached have to be frozen.

If they are frozen, you are already able to calculate embeddings only once per training.

Labels caching has more strict rules:

1. All encoders have to be frozen. If at least one is not frozen, we can't cache labels.
2. Multiprocessing is not allowed.
3. Key extraction is not overridden.

Multiprocessing
---------------

Labels are stored in a dataset instance.
This instance, and therefore the label cache, is bound to the process in which it was created.
If we use multiprocessing, then the label cache is filled in a child process.
We simply don't have access to our label cache from the parent process during training, which makes it impossible to use multiprocessing in this case.

Speaking about the code, this limitation requires `num_workers=None` (default value).

Key extractor
-------------

The key extractor is the function used to get the key for the entry we want to store in the cache.
By default, `key_extractor` uses the index of the item in the dataset as the cache key.
This is usually sufficient, however it has its drawbacks that you may want to avoid.

For instance, in some cases data-independent keys may not be acceptable or desirable, and you may want to establish such a connection between them.

You can provide custom ``key_extractors`` and extract keys from features in your own way to obtain desired properties.
(Here, by features we mean raw data written into a corresponding ``obj`` field in ``SimilaritySample``)

If you're using a custom key extractor, you'll need to access the features during training to get the key from it.
But retrieving features from a dataset is exactly what we wanted to avoid when caching labels.
Hence, usage of a custom key extractor makes labels caching meaningless.

.. code-block:: python
    :caption: provide custom key extractor

    def configure_caches(self) -> Optional[CacheConfig]:
        def custom_key_extractor(feature):
            return feature['filename']  # let's assume we have a dict as a feature

        return CacheConfig(
                key_extractor=custom_key_extractor  # use feature's filename as a key
            )


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