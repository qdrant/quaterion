import os
import pickle
from typing import Callable

import numpy as np
import tqdm
from pytorch_lightning import seed_everything
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from quaterion.dataset import GroupSimilarityDataLoader, SimilarityGroupSample

# set seed to deterministically sample train and test categories later on
seed_everything(seed=42)

# dataset will be downloaded to this directory under local directory
dataset_path = os.path.join(".", "torchvision", "datasets")


def get_raw_dataset(input_size: int, split_cache_path="split_cache.pkl"):
    """
    Create dataset for extracting images, associated with vectors.
    Args:
        input_size: Resize images to this size
        split_cache_path: Path to train split

    Returns:

    """
    transform = transforms.Compose(
        [
            transforms.Resize(input_size, max_size=input_size + 1),
        ]
    )

    full_dataset = datasets.StanfordCars(
        root=dataset_path, split="train", download=True, transform=transform
    ) + datasets.StanfordCars(
        root=dataset_path, split="test", download=True, transform=transform
    )

    # Use same indexes, as was used for training
    train_indices, test_indices = pickle.load(open(split_cache_path, "rb"))

    train_dataset = Subset(full_dataset, train_indices)

    test_dataset = Subset(full_dataset, test_indices)

    return train_dataset, test_dataset


def get_datasets(
    input_size: int,
    split_cache_path="split_cache.pkl",
):
    # Use Mean and std values for the ImageNet dataset as the base model was pretrained on it.
    # taken from https://www.geeksforgeeks.org/how-to-normalize-images-in-pytorch/
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # create train and test transforms
    train_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
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

    # we need to merge train and test splits into a full dataset first,
    # and then we will split it to two subsets again with each one composed of distinct labels.
    full_dataset = datasets.StanfordCars(
        root=dataset_path, split="train", download=True
    ) + datasets.StanfordCars(root=dataset_path, split="test", download=True)

    train_indices, test_indices = None, None

    if not split_cache_path or not os.path.exists(split_cache_path):
        # full_dataset contains examples from 196 categories labeled with an integer from 0 to 195
        # randomly sample half of it to be used for training
        train_categories = np.random.choice(a=196, size=196 // 2, replace=False)

        # get a list of labels for all samples in the dataset
        labels_list = np.array([label for _, label in tqdm.tqdm(full_dataset)])

        # get a mask for indices where label is included in train_categories
        labels_mask = np.isin(labels_list, train_categories)

        # get a list of indices to be used as train samples
        train_indices = np.argwhere(labels_mask).squeeze()

        # others will be used as test samples
        test_indices = np.argwhere(np.logical_not(labels_mask)).squeeze()

    if train_indices is None or test_indices is None:
        train_indices, test_indices = pickle.load(open(split_cache_path, "rb"))
    else:
        pickle.dump((train_indices, test_indices), open(split_cache_path, "wb"))

    # now that we have distinct indices for train and test sets, we can use `Subset` to create new datasets
    # from `full_dataset`, which contain only the samples at given indices.
    # finally, we apply transformations created above.
    train_dataset = CarsDataset(
        Subset(full_dataset, train_indices), transform=train_transform
    )

    test_dataset = CarsDataset(
        Subset(full_dataset, test_indices), transform=test_transform
    )

    return train_dataset, test_dataset


def get_dataloaders(
    batch_size: int,
    input_size: int,
    shuffle: bool = False,
    split_cache_path="split_cache.pkl",
):
    train_dataset, test_dataset = get_datasets(input_size, split_cache_path)

    train_dataloader = GroupSimilarityDataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )

    test_dataloader = GroupSimilarityDataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
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
