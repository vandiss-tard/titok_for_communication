from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from titok.Bin_transmission import bits_per_token, indices_to_bytes
from titok.blocks import TiTok
from titok.config import TiTokConfig


# ============================================================
# 文件路径与运行参数
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent

CHECKPOINT_PATH = PROJECT_ROOT / "runs/train/best.pt"
INPUT_IMAGE_PATH = PROJECT_ROOT / "data/train/dive_3_0063.jpg"
INDEX_OUTPUT_PATH = PROJECT_ROOT / "pre_transmit_data/dive_3_0063_indices.bin"
DEVICE = "auto"


def resolve_device() -> torch.device:
    if DEVICE == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    return torch.device(DEVICE)


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> TiTok:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    config = TiTokConfig(**checkpoint["config"])
    config.validate()

    model = TiTok(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def load_image(
    image_path: Path,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB"))

    return tensor.unsqueeze(0).to(device)


def main() -> None:
    device = resolve_device()
    model = load_model(CHECKPOINT_PATH, device)
    config = model.config

    image = load_image(
        INPUT_IMAGE_PATH,
        config.image_size,
        device,
    )

    indices = model.encode(image)
    packed = indices_to_bytes(
        indices,
        codebook_size=config.codebook_size,
    )

    expected_bytes = (
        config.latent_tokens
        * bits_per_token(config.codebook_size)
        + 7
    ) // 8

    if packed.shape[-1] != expected_bytes:
        raise RuntimeError(
            f"Expected {expected_bytes} bytes, "
            f"got {packed.shape[-1]}"
        )

    INDEX_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_OUTPUT_PATH.write_bytes(
        bytes(packed.flatten().cpu().tolist())
    )

    print(f"input image  : {INPUT_IMAGE_PATH}")
    print(f"image shape  : {tuple(image.shape)}")
    print(f"indices      : {tuple(indices.shape)}")
    print(f"index range  : {indices.min().item()} ~ {indices.max().item()}")
    print(f"unique codes : {torch.unique(indices).numel()}")
    print(f"binary file  : {INDEX_OUTPUT_PATH}")
    print(f"binary size  : {INDEX_OUTPUT_PATH.stat().st_size} Byte")


if __name__ == "__main__":
    main()
