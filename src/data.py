from collections import Counter
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .config import (
    BATCH_SIZE,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_WORKERS,
    PLANTDOC_CLASS_MAPPING,
    VALID_IMAGE_EXTENSIONS,
)
from .reproducibility import seed_worker


def build_transforms():
    """Return separate train/eval transforms so augmentation can be added safely later."""
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, eval_transform


def build_plantvillage_datasets(train_dir, val_dir):
    train_transform, eval_transform = build_transforms()

    train_data = datasets.ImageFolder(str(train_dir), transform=train_transform)
    val_data = datasets.ImageFolder(str(val_dir), transform=eval_transform)

    if train_data.classes != val_data.classes:
        raise ValueError("Train and validation class order does not match.")
    if train_data.class_to_idx != val_data.class_to_idx:
        raise ValueError("Train and validation class indices do not match.")

    return train_data, val_data


def build_dataloaders(train_data, val_data, seed: int, batch_size: int = BATCH_SIZE):
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


class PlantDocMappedDataset(Dataset):
    """PlantDoc split aligned to PlantVillage labels without pooling train/test."""

    def __init__(
        self,
        root,
        split,
        plantvillage_class_to_idx,
        transform=None,
        class_mapping=None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.class_mapping = class_mapping or PLANTDOC_CLASS_MAPPING
        self.samples = []

        for pv_class, plantdoc_class in self.class_mapping.items():
            if pv_class not in plantvillage_class_to_idx:
                raise KeyError(f"PlantVillage class not found: {pv_class}")

            folder = self.root / split / plantdoc_class
            if not folder.is_dir():
                continue

            label = plantvillage_class_to_idx[pv_class]
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
                    continue
                self.samples.append((path, label, pv_class, plantdoc_class))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, _, _ = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_plantdoc_datasets(root, class_to_idx):
    _, eval_transform = build_transforms()
    train_data = PlantDocMappedDataset(
        root=root,
        split="train",
        plantvillage_class_to_idx=class_to_idx,
        transform=eval_transform,
    )
    test_data = PlantDocMappedDataset(
        root=root,
        split="test",
        plantvillage_class_to_idx=class_to_idx,
        transform=eval_transform,
    )
    return train_data, test_data


def build_eval_loader(dataset, batch_size: int = BATCH_SIZE):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def class_counts_from_imagefolder(dataset):
    return Counter(label for _, label in dataset.samples)


def class_counts_from_plantdoc(dataset):
    return Counter(label for _, label, _, _ in dataset.samples)
