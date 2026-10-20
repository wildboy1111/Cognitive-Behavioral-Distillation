"""One trial-aligned HDF5 interface and explicit train/validation/test splits."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import ModelConfig

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class TrialData:
    """Aligned trials; EEG [N,C,T], ET [N,T,F], zero-based class labels.

    Signals are read once and HDF5 handles are closed before spawning workers.
    No resampling, normalization or label remapping happens implicitly.
    """

    def __init__(self, path, num_classes):
        with h5py.File(path, "r") as handle:
            self.images = handle["images"].asstr()[:].tolist()
            raw_labels = handle["labels"][:]
            if not np.issubdtype(raw_labels.dtype, np.integer):
                raise ValueError("labels must contain integers.")
            self.labels = raw_labels.astype(np.int64)
            self.eeg = handle["eeg"][:].astype(np.float32)
            self.et = handle["et"][:].astype(np.float32)
        n = len(self.images)
        if n == 0 or any(len(x) != n for x in (self.labels, self.eeg, self.et)):
            raise ValueError(
                "All modalities must contain the same nonzero trial count."
            )
        if self.labels.ndim != 1 or self.eeg.ndim != 3 or self.et.ndim != 3:
            raise ValueError("Expected labels [N], EEG [N,C,T], ET [N,T,F].")
        if self.labels.min() < 0 or self.labels.max() >= num_classes:
            raise ValueError("Labels must be zero-based and smaller than num_classes.")
        if not np.isfinite(self.eeg).all() or not np.isfinite(self.et).all():
            raise ValueError(
                "EEG and ET must be finite; clean missing values explicitly."
            )
        self.model_config = ModelConfig(
            num_classes=num_classes,
            eeg_channels=self.eeg.shape[1],
            eeg_samples=self.eeg.shape[2],
            et_samples=self.et.shape[1],
            et_features=self.et.shape[2],
        )

    def validate_images(self, root):
        root = Path(root)
        for name in set(self.images):
            path = root / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing image: {path}")


class TrialDataset(Dataset):
    """Load images on demand; student evaluation omits physiological signals."""

    def __init__(self, data, indices, image_root, image_only=False):
        self.data = data
        self.indices = list(indices)
        self.image_root = Path(image_root)
        self.image_only = image_only

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        with Image.open(self.image_root / self.data.images[i]) as image:
            image = IMAGE_TRANSFORM(image.convert("RGB"))
        item = {"image": image, "label": int(self.data.labels[i])}
        if not self.image_only:
            item["eeg"] = torch.from_numpy(self.data.eeg[i])
            item["et"] = torch.from_numpy(self.data.et[i])
        return item


def load_split(path, fold, data):
    """Reject invalid indices, omitted trials and overlapping split membership."""
    folds = json.loads(Path(path).read_text(encoding="utf-8"))["folds"]
    if not 0 <= fold < len(folds):
        raise ValueError(f"fold must be between 0 and {len(folds) - 1}.")
    split = folds[fold]
    n = len(data.labels)
    all_indices = []
    for name in ("train", "val", "test"):
        indices = split[name]
        if not indices or any(type(i) is not int or not 0 <= i < n for i in indices):
            raise ValueError(f"Invalid or empty {name} split.")
        all_indices.extend(indices)
    if len(set(all_indices)) != len(all_indices):
        raise ValueError("Split indices overlap or contain duplicates.")
    if set(all_indices) != set(range(n)):
        raise ValueError("Every trial must appear in exactly one split per fold.")
    return {name: split[name] for name in ("train", "val", "test")}
