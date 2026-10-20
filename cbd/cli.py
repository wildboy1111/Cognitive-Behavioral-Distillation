"""A single command-line workflow for teacher, student and image inference."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from .config import ExperimentConfig, ModelConfig
from .data import IMAGE_TRANSFORM, TrialData, TrialDataset, load_split
from .engine import (
    evaluate,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    seed_everything,
    train_epoch,
    write_json,
)
from .models import Student, Teacher


def build_data(config, fold):
    data = TrialData(config.data, config.num_classes)
    data.validate_images(config.images)
    model_config = replace(
        data.model_config,
        student_dropout=config.student_dropout,
        private_dropout=config.private_dropout,
    )
    return data, model_config, load_split(config.splits, fold, data)


def make_loader(config, data, indices, *, training=False, image_only=False):
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        TrialDataset(data, indices, config.images, image_only=image_only),
        batch_size=config.batch_size,
        shuffle=training,
        num_workers=config.workers,
        generator=generator,
        pin_memory=resolve_device(config.device).type == "cuda",
    )


def train(config, stage, fold, teacher_path=None):
    seed_everything(config.seed)
    device = resolve_device(config.device)
    data, model_config, split = build_data(config, fold)
    teacher = None
    if stage == "student":
        if not teacher_path:
            raise ValueError("Student training requires --teacher.")
        checkpoint = load_checkpoint(teacher_path)
        if checkpoint["stage"] != "teacher" or checkpoint["fold"] != fold:
            raise ValueError("Teacher checkpoint must belong to this stage/fold.")
        if checkpoint["split"] != split:
            raise ValueError("Teacher and student must use identical trial splits.")
        if ModelConfig(**checkpoint["model_config"]) != model_config:
            raise ValueError("Teacher checkpoint has a different model configuration.")
        if checkpoint["experiment"]["data"] != config.data:
            raise ValueError(
                "Teacher and student must use the same prepared data file."
            )
        teacher = Teacher(model_config, pretrained=False).to(device)
        teacher.load_state_dict(checkpoint["state_dict"])
        teacher.requires_grad_(False).eval()
        model = Student(model_config, pretrained=config.pretrained).to(device)
        weights, metric = config.student_losses, "img"
    else:
        model = Teacher(model_config, pretrained=config.pretrained).to(device)
        weights, metric = config.teacher_losses, config.teacher_metric
        if metric not in {"image_mean", "eeg", "et", "img_g", "img_l"}:
            raise ValueError(
                "teacher_metric must be image_mean, eeg, et, img_g or img_l."
            )
    loaders = {
        key: make_loader(
            config,
            data,
            indices,
            training=key == "train",
            image_only=stage == "student" and key != "train",
        )
        for key, indices in split.items()
    }
    output = Path(config.output) / stage / f"fold_{fold}"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config.to_dict())
    write_json(output / "split.json", split)
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_gamma)
    best_ba = float("-inf")
    with (output / "history.jsonl").open("w", encoding="utf-8") as history:
        for epoch in range(1, config.epochs + 1):
            loss = train_epoch(
                model, loaders["train"], optimizer, device, weights, teacher
            )
            validation = evaluate(model, loaders["val"], device, stage)
            if metric == "image_mean":
                score = (
                    sum(
                        validation[name]["balanced_accuracy"]
                        for name in ("img_g", "img_l")
                    )
                    / 2
                )
            else:
                score = validation[metric]["balanced_accuracy"]
            row = {"epoch": epoch, "train_loss": loss, "validation": validation}
            history.write(json.dumps(row) + "\n")
            history.flush()
            if score > best_ba:
                best_ba = score
                save_checkpoint(
                    output / "best.pt",
                    model,
                    model_config,
                    config,
                    stage,
                    fold,
                    split,
                    epoch,
                    best_ba,
                )
            scheduler.step()
            print(
                f"{stage} fold={fold} epoch={epoch}/{config.epochs} loss={loss:.4f} val_BA={score:.4f}",
                flush=True,
            )
    best = load_checkpoint(output / "best.pt")
    model.load_state_dict(best["state_dict"])
    # The held-out test set is only evaluated after choosing the best validation epoch.
    result = {
        "stage": stage,
        "fold": fold,
        "best_epoch": best["epoch"],
        "test": evaluate(model, loaders["test"], device, stage),
    }
    write_json(output / "metrics.json", result)
    print(json.dumps(result, indent=2))
    return result


def restore(path, device):
    checkpoint = load_checkpoint(path)
    model_config = ModelConfig(**checkpoint["model_config"])
    model = (Teacher if checkpoint["stage"] == "teacher" else Student)(
        model_config, pretrained=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def main():
    parser = argparse.ArgumentParser(description="Cognitive-Behavioral Distillation")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser(
        "train", help="Train a teacher or distill an image student"
    )
    training.add_argument("--config", default="configs/default.json")
    training.add_argument("--stage", choices=("teacher", "student"), required=True)
    training.add_argument("--fold", default="0", help="Zero-based fold number, or all")
    training.add_argument(
        "--teacher", help="Teacher checkpoint; supports {fold} in the path"
    )
    checking = commands.add_parser(
        "check-data", help="Validate aligned data, images and all folds"
    )
    checking.add_argument("--config", default="configs/default.json")
    testing = commands.add_parser(
        "evaluate", help="Evaluate the saved fold's test partition"
    )
    testing.add_argument("--config", default="configs/default.json")
    testing.add_argument("--checkpoint", required=True)
    prediction = commands.add_parser("predict", help="Image-only student inference")
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--image", required=True)
    prediction.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.command == "predict":
        device = resolve_device(args.device)
        model, checkpoint = restore(args.checkpoint, device)
        if checkpoint["stage"] != "student":
            raise ValueError("predict requires a student checkpoint.")
        with Image.open(args.image) as image:
            image = IMAGE_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            probabilities = model(image)["logits"]["img"].softmax(1)[0].cpu().tolist()
        print(
            json.dumps(
                {
                    "class": int(
                        max(range(len(probabilities)), key=probabilities.__getitem__)
                    ),
                    "probabilities": probabilities,
                }
            )
        )
        return
    config = ExperimentConfig.load(args.config)
    if args.command == "check-data":
        data = TrialData(config.data, config.num_classes)
        data.validate_images(config.images)
        folds = json.loads(Path(config.splits).read_text(encoding="utf-8"))["folds"]
        if not folds:
            raise ValueError("At least one split is required.")
        for fold in range(len(folds)):
            split = load_split(config.splits, fold, data)
            print(
                f"fold={fold}: " + ", ".join(f"{k}={len(v)}" for k, v in split.items())
            )
        print(data.model_config)
    elif args.command == "train":
        folds = json.loads(Path(config.splits).read_text(encoding="utf-8"))["folds"]
        chosen = range(len(folds)) if args.fold == "all" else [int(args.fold)]
        if not folds:
            raise ValueError("At least one split is required.")
        results = []
        for fold in chosen:
            teacher_path = args.teacher.format(fold=fold) if args.teacher else None
            results.append(train(config, args.stage, fold, teacher_path))
        write_json(Path(config.output) / f"{args.stage}_summary.json", results)
    elif args.command == "evaluate":
        device = resolve_device(config.device)
        model, checkpoint = restore(args.checkpoint, device)
        data, model_config, split = build_data(config, checkpoint["fold"])
        if (
            model_config != ModelConfig(**checkpoint["model_config"])
            or split != checkpoint["split"]
        ):
            raise ValueError(
                "Evaluation data configuration/split differs from the checkpoint."
            )
        loader = make_loader(
            config, data, split["test"], image_only=checkpoint["stage"] == "student"
        )
        print(
            json.dumps(evaluate(model, loader, device, checkpoint["stage"]), indent=2)
        )
