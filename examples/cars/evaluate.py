import argparse
import os
import torch
import torch.nn as nn
import torchvision
from quaterion_models import MetricModel
from quaterion_models.heads import EmptyHead

from examples.cars.config import IMAGE_SIZE
from examples.cars.encoders import CarsEncoder
from quaterion import Quaterion
from quaterion.eval.evaluator import Evaluator
from quaterion.eval.group import RetrievalRPrecision
from quaterion.eval.samplers.group_sampler import GroupSampler
from .data import get_datasets

BATCH_SIZE = 32
METRIC_BATCH = 1024


def eval_base_encoder(dataset, device):
    print("Evaluating base encoder...")
    base_encoder = torchvision.models.resnet152(pretrained=True)
    base_encoder.fc = nn.Identity()

    cars_encoder = CarsEncoder(base_encoder)
    cars_encoder.to(device=device)
    cars_encoder.eval()

    result = Quaterion.evaluate(
        evaluator=Evaluator(
            metrics=RetrievalRPrecision(),
            sampler=GroupSampler(sample_size=1000, device=device),
        ),
        model=MetricModel(
            encoders=cars_encoder, head=EmptyHead(cars_encoder.embedding_size)
        ),
        dataset=dataset,
    )

    print(result)


def eval_tuned_encoder(dataset, device):
    print("Evaluating tuned encoder...")
    tuned_cars_model = MetricModel.load(
        os.path.join(os.path.dirname(__file__), "best_cars_encoders")
    ).to(device)
    tuned_cars_model.eval()

    result = Quaterion.evaluate(
        evaluator=Evaluator(
            metrics=RetrievalRPrecision(),
            sampler=GroupSampler(sample_size=1000, device=device),
        ),
        model=tuned_cars_model,
        dataset=dataset,
    )

    print(result)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="both",
        choices=("both", "base", "tuned"),
        help="Model to evaluate",
    )

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Preparing test data loader...")

    _, test_dataset = get_datasets(
        input_size=IMAGE_SIZE,
        split_cache_path="split_cache.pkl",  # reuse existing database split
    )

    if args.model == "base":
        eval_base_encoder(test_dataset, device)
    elif args.model == "tuned":
        eval_tuned_encoder(test_dataset, device)
    else:
        eval_base_encoder(test_dataset, device)
        eval_tuned_encoder(test_dataset, device)
