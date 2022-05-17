"""Prepare vectors for serving
"""
import json

import PIL.Image
import argparse
import numpy as np
import os
import torch
import tqdm
from quaterion_models import SimilarityModel
from quaterion_models.heads import EncoderHead, EmptyHead

from examples.cars.config import IMAGE_SIZE
from examples.cars.data import get_dataloaders, get_raw_dataset
from examples.cars.models import Model

BATCH_SIZE = 32


class EvalModel(Model):
    def configure_head(self, input_embedding_size) -> EncoderHead:
        return EmptyHead(input_embedding_size=input_embedding_size)


def eval_model(dataloader, model):
    embeddings = []

    for i, (_, images, labels) in enumerate(tqdm.tqdm(dataloader)):
        with torch.no_grad():
            embeddings_batch = model.encode(
                images, batch_size=BATCH_SIZE, to_numpy=True
            )
            embeddings.append(embeddings_batch)

    return np.concatenate(embeddings)


def serve_pretrained_embeddings(dataloader):
    model = EvalModel(lr=0, mining="all").model
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return eval_model(dataloader, model)


def serve_tuned_embeddings(dataloader, model_path):
    model = SimilarityModel.load(model_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return eval_model(dataloader, model)


def save_images(dataset, save_dir):
    labels_path = os.path.join(save_dir, "labels.jsonl")
    images_path = os.path.join(save_dir, "imgs")
    with open(labels_path, "w") as out:
        for image_id in tqdm.tqdm(range(0, len(dataset))):
            sample = dataset[image_id]
            image: PIL.Image.Image = sample[0]
            label = sample[1]
            out.write(json.dumps({"label": label}))
            out.write("\n")
            image.save(
                open(os.path.join(images_path, f"{image_id}.jpg"), "wb"), format="jpeg"
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tuned-model",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "cars_encoders"),
        help="Path to tuned model",
    )
    ap.add_argument(
        "--save-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "cars_embeddings"),
        help="Where to save the servable embeddings",
    )
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    _, test_dataset = get_raw_dataset(input_size=IMAGE_SIZE)
    save_images(test_dataset, args.save_dir)

    print("Preparing test data loader...")
    _, test_dl = get_dataloaders(
        batch_size=BATCH_SIZE, shuffle=False, input_size=IMAGE_SIZE
    )

    embeddings = serve_pretrained_embeddings(test_dl)

    np.save(os.path.join(args.save_dir, "original.npy"), embeddings, allow_pickle=False)

    embeddings = serve_tuned_embeddings(test_dl, args.tuned_model)

    np.save(os.path.join(args.save_dir, "tuned.npy"), embeddings, allow_pickle=False)
