import torch
import torch.nn.functional as F
from torch import Tensor, LongTensor

class NPairLoss(torch.nn.Module):
    def __init__(self, margin: float = 0.1):
        """
        N-Pair loss functionm for learning embeddings.
        
        Args:
        - margin (float) : Margin value for the loss calculation
        """

        super(NPairLoss, self).__init__()
        self.margin = margin

    
    def forward(self, embeddings: Tensor, labels: LongTensor) -> Tensor:
        """
        Calculate the N-Pair loss given a batch of embeddings and their labels.
        Args:
        - embeddings (Tensor): Batch of embeddings.
        - labels (LongTensor): Corresponding labels indicating pairs.
        
        Returns:
        - Tensor: Computed loss value.
        """

        # For pairs, check if the number of samples is even
        if embeddings.size(0) % 2!= 0:
            raise ValueError("Number of samples must be even for N-Pair loss.")
        
        # Split the embeddings into anchor -positive pairs
        anchors = embeddings[::2]
        positives = embeddings[1::2]

        # Calculate the pairwise cosine similarities
        similarities = F.cosine_similarity(anchors, positives)

        # Reshape labels for comparison
        labels = labels.view(-1,2)

        # Generate mask for positive pairs
        mask = torch.eq(labels[:,0], labels[:,1]).float()

        # Loss calculation for positive pairs
        loss_positives = -torch.log(similarities + 1e-8) * mask

        # Calculate the negative part of loss
        similarities_matrix = similarities.view(-1,1).repeat(1, embeddings.size(0)//2)
        similarities_matrix = similarities_matrix.view(-1, embeddings.size(0) // 2)

        # Mask for negative pairs
        mask_negative = 1 - mask

        # Maximum similarity excluding the positive pair for each other
        max_similarity =  (similarities_matrix * mask_negative).max(dim=1)[0]

        # Loss calculation for negative part
        loss_negative = torch.relu(self.margin - max_similarity + similarities)

        # Compute final loss
        loss = (loss_positives + loss_negative).mean()

        return loss
