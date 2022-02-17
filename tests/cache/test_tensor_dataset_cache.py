import os.path

from torchvision import transforms
from torchvision.datasets import MNIST

from quaterion.dataset import GroupSimilarityDataLoader
from quaterion.dataset.similarity_dataset import SimilarityGroupDataset


def test_tensor_dataset_cache():
    tmp_dir_name = os.path.join(os.path.dirname(__file__), 'data', 'mnist')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = SimilarityGroupDataset(MNIST(tmp_dir_name, download=True, transform=transform))
    dataloader = GroupSimilarityDataLoader(dataset)

    for batch in dataloader:
        print(batch)
        break
