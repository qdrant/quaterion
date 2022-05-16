import argparse
import numpy as np
import os

import torch
import torch.nn as nn
import torchvision
import tqdm

from examples.cars.config import IMAGE_SIZE
from examples.cars.encoders import CarsEncoder
from quaterion import Quaterion
from quaterion.eval.evaluator import Evaluator
from quaterion.eval.group import RetrievalRPrecision
from quaterion_models import MetricModel

from quaterion_models.encoders import Encoder

from quaterion_models.heads import EmptyHead

from quaterion.eval.samplers.group_sampler import GroupSampler
from .data import get_dataloaders

BATCH_SIZE = 32
METRIC_BATCH = 1024


def eval_base_encoder(test_dl, device):
    print("Evaluating base encoder...")
    base_encoder = torchvision.models.resnet152(pretrained=True)
    base_encoder.fc = nn.Identity()

    cars_encoder = CarsEncoder(base_encoder)
    cars_encoder.to(device=device)
    cars_encoder.eval()

    Quaterion.evaluate(
        evaluator=Evaluator(
            metrics=RetrievalRPrecision(),
            sampler=GroupSampler(
                sample_size=1000,
                device=device
            )
        ),
        model=MetricModel(
            encoders=cars_encoder,
            head=EmptyHead(cars_encoder.embedding_size)
        ),
        dataset=test_dl.dataset
    )


def eval_tuned_encoder(test_dl, device):
    print("Evaluating tuned encoder...")
    tuned_encoder = MetricModel.load(
        os.path.join(os.path.dirname(__file__), "cars_encoders")
    ).to(device)
    tuned_encoder.eval()

    all_metrics = []
    num_metric_batches = 0
    metric = RetrievalRPrecision()

    for i, (_, images, labels) in enumerate(tqdm.tqdm(test_dl)):
        with torch.no_grad():
            # images = torch.stack(images).to(device)
            embeddings = tuned_encoder.encode(
                images, batch_size=BATCH_SIZE, to_numpy=False
            )
            metric.update(embeddings, labels["groups"].to(device))
            num_metric_batches += 1

            if METRIC_BATCH / BATCH_SIZE <= num_metric_batches:
                running_score = float(metric.evaluate())
                all_metrics.append(running_score)
                print(f"Running score at step {i}: {running_score:.4f}")
                metric.reset()
                num_metric_batches = 0

    final_score = np.mean(all_metrics)
    print(f"Final Retrieval R-Precision score for tuned encoder: {final_score:.4f}")


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
    _, test_dl = get_dataloaders(
        batch_size=BATCH_SIZE, shuffle=False, input_size=IMAGE_SIZE
    )

    if args.model == "base":
        eval_base_encoder(test_dl, device)
    elif args.model == "tuned":
        eval_tuned_encoder(test_dl, device)
    else:
        eval_base_encoder(test_dl, device)
        eval_tuned_encoder(test_dl, device)
