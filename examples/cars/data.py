import os
from typing import Callable

import numpy as np
from quaterion.dataset import (
    GroupSimilarityDataLoader,
    SimilarityGroupSample,
)
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


# set seed to deterministically sample train and test categories later on
np.random.seed(42)


def get_dataloaders(
    batch_size: int = 32,
    input_size: int = 128,
    shuffle: bool = False,
):
    # Use Mean and std values for the ImageNet dataset as the base model was pretrained on it.
    # taken from https://www.geeksforgeeks.org/how-to-normalize-images-in-pytorch/
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # dataset will be downloaded to this directory under local directory
    path = os.path.join(".", "torchvision", "datasets")

    # we need to merge train and test splits into a full dataset first,
    # and then we will split it to two subsets again with each one composed of distinct labels.
    full_dataset = datasets.StanfordCars(
        root=path, split="train", download=True
    ) + datasets.StanfordCars(root=path, split="test", download=True)

    # full_dataset contains examples from 196 categories labeled with an integer from 0 to 195
    # randomly sample half of it to be used for training
    train_categories = np.random.choice(a=196, size=196 // 2, replace=False)

    # get a list of labels for all samples in the dataset
    labels_list = np.array([label for _, label in full_dataset])

    # get a mask for indices where label is included in train_categories
    labels_mask = np.isin(labels_list, train_categories)

    # get a list of indices to be used as train samples
    train_indices = np.argwhere(labels_mask).squeeze()

    # others will be used as test samples
    test_indices = np.argwhere(np.logical_not(labels_mask)).squeeze()

    # create train and test transforms
    train_transform = transforms.Compose(
        [
            # transforms.Resize((input_size, input_size)),
            transforms.RandomResizedCrop((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # now that we have distinct indices for train and test sets, we can use `Subset` to create new datasets
    # from `full_dataset`, which contain only the samples at given indices.
    # finally, we apply transformations created above.
    train_dataset = CarsDataset(
        Subset(full_dataset, train_indices), transform=train_transform
    )

    test_dataset = CarsDataset(
        Subset(full_dataset, test_indices), transform=test_transform
    )

    train_dataloader = GroupSimilarityDataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4
    )

    test_dataloader = GroupSimilarityDataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_dataloader, test_dataloader


class CarsDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Callable):
        self._dataset = dataset
        self._transform = transform

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index) -> SimilarityGroupSample:
        image, label = self._dataset[index]
        image = self._transform(image)

        return SimilarityGroupSample(obj=image, group=label)
