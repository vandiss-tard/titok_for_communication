from __future__ import annotations

from pathlib import Path

import torch
from torchvision.utils import save_image

from titok.Bin_transmission import bits_per_token, bytes_to_indices
from titok.blocks import TiTok
from titok.config import TiTokConfig


# ============================================================
# 文件路径与运行参数
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent

CHECKPOINT_PATH = PROJECT_ROOT / "runs/train/best.pt"
INDEX_INPUT_PATH = PROJECT_ROOT / "pre_transmit_data/dive_3_0063_indices.bin"
RECONSTRUCTION_OUTPUT_PATH = PROJECT_ROOT / "After_transmit_data/dive_3_0063_reconstruction.png"

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


def main() -> None:
    if not INDEX_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Index file not found: {INDEX_INPUT_PATH}"
        )

    device = resolve_device()
    model = load_model(CHECKPOINT_PATH, device)
    config = model.config

    raw_bytes = INDEX_INPUT_PATH.read_bytes()

    expected_bytes = (
        config.latent_tokens
        * bits_per_token(config.codebook_size)
        + 7
    ) // 8

    if len(raw_bytes) != expected_bytes:
        raise ValueError(
            f"Expected {expected_bytes} bytes, "
            f"got {len(raw_bytes)}"
        )

    data = torch.tensor(
        list(raw_bytes),
        dtype=torch.uint8,
    ).unsqueeze(0)

    indices = bytes_to_indices(
        data,
        num_tokens=config.latent_tokens,
        codebook_size=config.codebook_size,
    ).to(device)

    if indices.max().item() >= config.codebook_size:
        raise ValueError("Decoded index exceeds codebook size")

    reconstruction = model.decode(indices)

    RECONSTRUCTION_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    save_image(
        reconstruction.clamp(0.0, 1.0).cpu(),
        RECONSTRUCTION_OUTPUT_PATH,
    )

    print(f"index file   : {INDEX_INPUT_PATH}")
    print(f"file size    : {len(raw_bytes)} Byte")
    print(f"indices      : {tuple(indices.shape)}")
    print(f"index range  : {indices.min().item()} ~ {indices.max().item()}")
    print(f"unique codes : {torch.unique(indices).numel()}")
    print(f"output image : {RECONSTRUCTION_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
