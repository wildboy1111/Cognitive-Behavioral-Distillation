"""Shared optimization, evaluation and portable checkpoints."""

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.nn import functional as F


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def objective(output, labels, weights):
    """Sum branch cross-entropies and explicitly configured auxiliary terms.

    The aggregate loss_distill is diagnostic; weighting it together with its
    four components would count the same terms twice and is rejected.
    """
    unknown = set(weights) - set(output["loss"])
    if unknown:
        raise ValueError(f"Unknown loss terms: {sorted(unknown)}")
    if weights.get("loss_distill", 0) and any(
        value and key.startswith("loss_distill_") for key, value in weights.items()
    ):
        raise ValueError("Weight either loss_distill or its components, not both.")
    loss = sum(F.cross_entropy(logits, labels) for logits in output["logits"].values())
    return loss + sum(weight * output["loss"][name] for name, weight in weights.items())


def train_epoch(model, loader, optimizer, device, weights, teacher=None):
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        image, eeg, et, labels = [
            batch[key].to(device) for key in ("image", "eeg", "et", "label")
        ]
        optimizer.zero_grad(set_to_none=True)
        if teacher is None:
            output = model(image, eeg, et, labels)
        else:
            output = model(image, eeg, et, labels, teacher=teacher)
        loss = objective(output, labels, weights)
        if not torch.isfinite(loss):
            raise FloatingPointError("Training produced a non-finite loss.")
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
        count += len(labels)
    return total / count


@torch.no_grad()
def evaluate(model, loader, device, stage):
    model.eval()
    targets = []
    predictions = {}
    for batch in loader:
        image = batch["image"].to(device)
        if stage == "student":
            output = model(image)
        else:
            output = model(image, batch["eeg"].to(device), batch["et"].to(device))
        targets.extend(batch["label"].tolist())
        for name, logits in output["logits"].items():
            predictions.setdefault(name, []).extend(logits.argmax(1).cpu().tolist())
    return {
        name: {
            "accuracy": float(accuracy_score(targets, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(targets, pred)),
            "macro_f1": float(
                f1_score(targets, pred, average="macro", zero_division=0)
            ),
        }
        for name, pred in predictions.items()
    }


def save_checkpoint(
    path, model, model_config, experiment, stage, fold, split, epoch, metric
):
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "model_config": asdict(model_config),
            "experiment": experiment.to_dict(),
            "stage": stage,
            "fold": fold,
            "split": split,
            "epoch": epoch,
            "validation_ba": metric,
        },
        path,
    )


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
