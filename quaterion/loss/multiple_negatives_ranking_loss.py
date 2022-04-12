from typing import Type, Dict, Any

import torch

from torch import Tensor, LongTensor
import torch.nn.functional as F

from quaterion.distances import Distance
from quaterion.loss.pairwise_loss import PairwiseLoss


class MultipleNegativesRankingLoss(PairwiseLoss):
    """Implement Multiple Negatives Ranking Loss as described in
    https://arxiv.org/pdf/1705.00652.pdf

    This loss function works only with positive pairs, e.g., an `anchor` and a `positive`.
    For each pair, it uses `positive` of other pairs in the batch as negatives, so you don't need
    to worry about specifying negative examples. It is great for retrieval tasks such as
    question-answer retrieval, duplicate sentence retrieval, and cross-modal retrieval.
    It accepts pairs of anchor and positive embeddings to calculate a similarity matrix between
    them.
    Then, it minimizes negative log-likelihood for softmax-normalized similarity scores.
    This optimizes  retrieval of the correct positive pair when an anchor given.

    Note:
        :attr:`~quaterion.dataset.similarity_samples.SimilarityPairSample.score` and
        :attr:`~quaterion.dataset.similarity_samples.SimilarityPairSample.subgroup` values are
        ignored for this loss, assuming
        :attr:`~quaterion.dataset.similarity_samples.SimilarityPairSample.obj_a` and
        :attr:`~quaterion.dataset.similarity_samples.SimilarityPairSample.obj_b` form a positive
        pair, e.g., `label = 1`.

    Args:
        scale: Scaling value for multiplying with similarity scores to make cross-entropy work.
        distance_metric_name: Name of the metric to calculate similarities between embeddings,
            e.g., :class:`~quaterion.distances.Distance`.
            Optional, defaults to :attr:`~quaterion.distances.Distance.COSINE`.
            If :attr:`~quaterion.distances.Distance.DOT_PRODUCT`, `scale` must be `1`.
        symmetric: If True, loss is symmetric,
            i.e., it also accounts for retrieval of the correct anchor when a positive given.
    """

    def __init__(
        self,
        scale: float = 20.0,
        distance_metric_name: Distance = Distance.COSINE,
        symmetric: bool = False,
    ):
        super().__init__(distance_metric_name=distance_metric_name)
        self._scale = scale
        self._symmetric = symmetric

    def get_config_dict(self) -> Dict[str, Any]:
        """Config used in saving and loading purposes.

        Config object has to be JSON-serializable.

        Returns:
            Dict[str, Any]: JSON-serializable dict of params
        """
        config = self.get_config_dict()
        config.update(
            {
                "scale": self._scale,
                "symmetric": self._symmetric,
            }
        )

        return config

    def forward(
        self,
        embeddings: Tensor,
        pairs: LongTensor,
        labels: Tensor,
        subgroups: Tensor,
        **kwargs,
    ) -> Tensor:
        """Compute loss value.

        Args:
            embeddings: Batch of embeddings, first half of embeddings are embeddings
                of first objects in pairs, second half are embeddings of second objects
                in pairs.
            pairs: Indices of corresponding objects in pairs.
            labels: Ignored for this loss. Labels will be automatically formed from `pairs`.
            subgroups: Ignored for this loss.

            **kwargs: Additional key-word arguments for generalization of loss call

        Returns:
            Tensor: Scalar loss value
        """
        _warn = (
            "You seem to be using non-positive pairs. "
            "Make sure that `SimilarityPairSample.obj_a` and `SimilarityPairSample.obj_b` "
            "are positive pairs with a score of `1`"
        )
        assert labels is None or labels.sum() == labels.size()[0], _warn
        rep_anchor = embeddings[pairs[:, 0]]
        rep_positive = embeddings[pairs[:, 1]]

        # get similarity matrix to be used as logits
        # shape: (batch_size, batch_size)
        logits = self.distance_metric.similarity_matrix(rep_anchor, rep_positive)
        logits *= self._scale

        # create integer label IDs
        labels = torch.arange(
            start=0, end=logits.size()[0], dtype=torch.long, device=logits.device
        )

        # calculate loss
        loss = F.cross_entropy(logits, labels)

        if self._symmetric:
            loss += F.cross_entropy(logits.transpose(0, 1), labels)
            loss /= 2

        return loss
