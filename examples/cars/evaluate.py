import argparse
import os

import torch
import torch.nn as nn
import torchvision
from quaterion.eval.group import RetrievalRPrecision
from quaterion_models import MetricModel

from .data import get_dataloaders


def eval_base_encoder(test_dl, device):
    print("Evaluating base encoder...")
    base_encoder = torchvision.models.resnet18(pretrained=True)
    base_encoder.fc = nn.Identity()
    base_encoder = base_encoder.to(device)
    base_encoder.eval()

    metric = RetrievalRPrecision()

    for i, (_, images, labels) in enumerate(test_dl):
        with torch.no_grad():
            images = torch.stack(images).to(device)
            embeddings = base_encoder(images)
            metric.update(embeddings, labels["groups"])

            running_score = float(metric.compute())
            print(f"Running score at step {i}: {running_score:.4f}")

    final_score = float(metric.compute())
    print(f"Final Retrieval R-Precision score for base encoder: {final_score:.4f}")


def eval_tuned_encoder(test_dl, device):
    print("Evaluating tuned encoder...")
    tuned_encoder = MetricModel.load(
        os.path.join(os.path.dirname(__file__), "cars_encoders")
    ).to(device)
    tuned_encoder.eval()

    metric = RetrievalRPrecision()

    for i, (_, images, labels) in enumerate(test_dl):
        with torch.no_grad():
            # images = torch.stack(images).to(device)
            embeddings = tuned_encoder.encode(images, batch_size=384, to_numpy=False)
            metric.update(embeddings, labels["groups"])

            running_score = float(metric.compute())
            print(f"Running score at step {i}: {running_score:.4f}")

    final_score = float(metric.compute())
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
    test_dl, _ = get_dataloaders(batch_size=384, shuffle=False)

    if args.model == "base":
        eval_base_encoder(test_dl, device)
    elif args.model == "tuned":
        eval_tuned_encoder(test_dl, device)
    else:
        eval_base_encoder(test_dl, device)
        eval_tuned_encoder(test_dl, device)
