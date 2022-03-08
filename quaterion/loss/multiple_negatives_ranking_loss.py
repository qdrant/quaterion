from typing import Type, Dict, Any

import torch

from torch import Tensor, LongTensor
import torch.nn.functional as F

from quaterion.loss.metrics import SiameseDistanceMetric
from quaterion.loss.pairwise_loss import PairwiseLoss


class MultipleNegativesRankingLoss(PairwiseLoss):
    """Implement Multiple Negatives Ranking Loss as described in https://arxiv.org/pdf/1705.00652.pdf

    This loss function works only with positive pairs, and it uses non-pair samples in a batch
    as negatives. It is great for retrieval tasks such as question-answer retrieval,
    duplicate sentence retrieval, and cross-modal retrieval.
    It accepts pairs of anchor and positive embeddings to calculate a similarity matrix between them.
    Then, it minimizes negative log-likelihood for softmax-normalized similarity scores.
    This optimizes  retrieval of the correct positive pair when an anchor given.


    Args:
        scale: Scaling value for multiplying with similarity scores to make cross-entropy work.
        similarity_metric_name: Name of the metric to calculate similarities between embeddings.
            Must be either `"cosine"` or `"dot_product"`. If `"dot_product"`, `scale` must be `1`.
        symmetric: If True, loss is symmetric,
            i.e., it also accounts for retrieval of the correct anchor when a positive given.
    """

    @classmethod
    def metric_class(cls) -> Type:
        """Class with metrics available for current loss.

        Returns:
            Type: class containing metrics
        """
        return SiameseDistanceMetric

    def __init__(
        self,
        scale: float = 20.0,
        similarity_metric_name: str = "cosine",
        symmetric: bool = False,
    ):
        similarity_metrics = ["cosine", "dot_product"]
        if similarity_metric_name not in similarity_metrics:
            raise ValueError(
                f"Not supported similarity metric for this loss: {similarity_metric_name}. "
                f"Must be one of {', '.join(similarity_metrics)}"
            )

        super().__init__(distance_metric_name=similarity_metric_name + "_distance")
        self._scale = scale
        self._similarity_metric_name = similarity_metric_name
        self._symmetric = symmetric

    def get_config_dict(self) -> Dict[str, Any]:
        """Config used in saving and loading purposes.

        Config object has to be JSON-serializable.

        Returns:
            Dict[str, Any]: JSON-serializable dict of params
        """
        return {
            "scale": self._scale,
            "similarity_metric_name": self._similarity_metric_name,
            "symmetric": self._symmetric,
        }

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
            labels: Scores of positive and negative objects.
            subgroups: subgroups to distinguish objects which can and cannot be used
                as negative examples
            **kwargs: additional key-word arguments for generalization of loss call

        Returns:
            Tensor: Scalar loss value
        """
        rep_anchor = embeddings[pairs[:, 0]]
        rep_positive = embeddings[pairs[:, 1]]

        # shape: (batch_size, batch_size)
        distance_matrix = self.distance_metric(rep_anchor, rep_positive, matrix=True)

        # convert distance values to similarity scores and then scale to use as logits
        logits = (
            1 - distance_matrix
            if self._similarity_metric_name == "cosine"
            else -distance_matrix
        )
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
