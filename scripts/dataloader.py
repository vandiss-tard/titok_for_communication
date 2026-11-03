from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


class ImageDirectory(Dataset):
    """递归读取图片，并对训练集和验证集使用不同变换。"""

    def __init__(
        self,
        root: Path,
        image_size: int,
        train: bool,
    ) -> None:
        self.files = sorted(
            path
            for path in root.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        if not self.files:
            raise FileNotFoundError(f"No images found in {root}")

        if train:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.RandomCrop(
                        (image_size, image_size)
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.CenterCrop(
                        (image_size, image_size)
                    ),
                    transforms.ToTensor(),
                ]
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.files[index]) as image:
            return self.transform(image.convert("RGB"))


def build_dataloaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    val_ratio: float,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    train_dataset = ImageDirectory(
        data_dir,
        image_size,
        train=True,
    )
    val_dataset = ImageDirectory(
        data_dir,
        image_size,
        train=False,
    )

    total_size = len(train_dataset)
    if total_size < 2:
        raise ValueError("At least two images are required")

    val_size = max(1, int(total_size * val_ratio))
    val_size = min(val_size, total_size - 1)
    train_size = total_size - val_size

    split_generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        total_size,
        generator=split_generator,
    )

    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()

    train_set = Subset(train_dataset, train_indices)
    val_set = Subset(val_dataset, val_indices)

    shuffle_generator = torch.Generator().manual_seed(seed + 1)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=shuffle_generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    info = {
        "total_size": total_size,
        "train_size": train_size,
        "val_size": val_size,
    }
    return train_loader, val_loader, info
