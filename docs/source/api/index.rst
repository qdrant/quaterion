API References
~~~~~~~~~~~~~~

DATA
----

Dataloaders
+++++++++++

.. py:currentmodule:: quaterion.dataset.similarity_data_loader

.. autosummary::
    :nosignatures:

    SimilarityDataLoader
    PairsSimilarityDataLoader
    GroupSimilarityDataLoader

Datasets
++++++++

.. py:currentmodule:: quaterion.dataset.similarity_dataset

.. autosummary::
    :nosignatures:

    SimilarityGroupDataset

Samples
+++++++

.. py:currentmodule:: quaterion.dataset.similarity_samples

.. autosummary::
    :nosignatures:

    SimilarityGroupSample
    SimilarityPairSample

DISTANCES
---------

.. py:currentmodule:: quaterion.distances

.. autosummary::
    :nosignatures:

    ~cosine.Cosine
    ~dot_product.DotProduct
    ~euclidean.Euclidean
    ~manhattan.Manhattan

EVAL
----

Counters
++++++++

.. py:currentmodule:: quaterion.eval

.. autosummary::
    :nosignatures:

    ~attached_metric.AttachedMetric
    ~evaluator.Evaluator

Group metrics
+++++++++++++

.. py:currentmodule:: quaterion.eval.group

.. autosummary::
    :nosignatures:

    ~group_metric.GroupMetric
    ~retrieval_r_precision.RetrievalRPrecision

Pair metrics
++++++++++++

.. py:currentmodule:: quaterion.eval.pair

.. autosummary::
    :nosignatures:

    ~pair_metric.PairMetric
    ~retrieval_precision.RetrievalPrecision
    ~retrieval_reciprocal_rank.RetrievalReciprocalRank


Samplers
++++++++

.. py:currentmodule:: quaterion.eval.samplers

.. autosummary::
    :nosignatures:

    ~group_sampler.GroupSampler
    ~pair_sampler.PairSampler


LOSSES
------

Base
++++

.. py:currentmodule:: quaterion.loss

.. autosummary::
    :nosignatures:

    ~group_loss.GroupLoss
    ~pairwise_loss.PairwiseLoss

Implementations
+++++++++++++++

.. py:currentmodule:: quaterion.loss

.. autosummary::
    :nosignatures:

    ~arcface_loss.ArcFaceLoss
    ~contrastive_loss.ContrastiveLoss
    ~multiple_negatives_ranking_loss.MultipleNegativesRankingLoss
    ~softmax_loss.SoftmaxLoss
    ~triplet_loss.TripletLoss
    ~circle_loss.CircleLoss
    ~fastap_loss.FastAPLoss
    ~cos_face_loss.CosFaceLoss

Extras
++++++

.. py:currentmodule:: quaterion.loss.extras

.. autosummary::
    :nosignatures:

    ~pytorch_metric_learning_wrapper.PytorchMetricLearningWrapper

MAIN
----

.. py:currentmodule:: quaterion.main

.. autosummary::
    :nosignatures:

    Quaterion

TRAIN
-----

TrainableModel
++++++++++++++

.. py:currentmodule:: quaterion.train.trainable_model

.. autosummary::
    :nosignatures:

    TrainableModel

Cache
+++++

.. py:currentmodule:: quaterion.train.cache

.. autosummary::
    :nosignatures:

    ~cache_config.CacheConfig
    ~cache_config.CacheType

UTILS
-----

.. py:currentmodule:: quaterion.utils

.. autosummary::
    :nosignatures:

    ~enums.TrainStage
    ~utils.get_triplet_mask
    ~utils.get_anchor_positive_mask
    ~utils.get_anchor_negative_mask
