import os

from torchvision import datasets

def get_dataloader(split: str = "train", batch_size: int = 128, input_size: int = 128, shuffle: bool = True):
    # Mean and std values taken from https://github.com/LJY-HY/cifar_pytorch-lightning/blob/master/datasets/CIFAR.py#L43
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    path = os.path.join(os.path.expanduser("~"), "torchvision", "datasets")

    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    dataset = SimilarityGroupDataset(
        datasets.CIFAR100(root=path, download=True, transform=transform)
    )
    dataloader = GroupSimilarityDataLoader(dataset, batch_size=128, shuffle=True)
    return dataloader

