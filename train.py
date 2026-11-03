from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import lpips
import torch
from torch.nn import functional as F

from scripts.dataloader import build_dataloaders
from titok.blocks import TiTok
from titok.config import TiTokConfig
from titok.discriminator import PatchDiscriminator


# ============================================================
# 训练参数（模型参数统一放在 titok/config.py）
# ============================================================
@dataclass(slots=True)
class TrainConfig:
    # 路径
    data_dir: Path = Path("data/train")
    output_dir: Path = Path("runs/train")

    # 训练
    batch_size: int = 1
    val_ratio: float = 0.2
    epochs: int = 20
    lr: float = 3e-4
    discriminator_lr: float = 1e-4
    weight_decay: float = 0.05

    # 损失权重
    recon_weight: float = 1.0
    perceptual_weight: float = 0.1
    gan_weight: float = 0.01
    vq_loss_weight: float = 1.0

    # GAN
    gan_start_epoch: int = 1

    # 训练稳定性
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    log_every: int = 1
    seed: int = 42
    device: str = "auto"


TRAIN_CONFIG = TrainConfig()


def compute_codebook_metrics(
    counts: torch.Tensor,
) -> dict[str, float]:
    """根据每个 code 的使用次数计算码本指标。"""
    codebook_size = counts.numel()
    total_tokens = counts.sum().item()

    if total_tokens == 0:
        return {
            "unique_codes": 0.0,
            "perplexity": 0.0,
            "dead_code_ratio": 1.0,
        }

    probabilities = counts.float() / total_tokens
    used = counts > 0
    used_count = used.sum().item()

    entropy = -(
        probabilities[used] * probabilities[used].log()
    ).sum()
    perplexity = entropy.exp().item()

    return {
        "unique_codes": float(used_count),
        "perplexity": perplexity,
        "dead_code_ratio": 1.0 - used_count / codebook_size,
    }


def compute_perceptual_loss(
    perceptual_model: lpips.LPIPS,
    reconstruction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    reconstruction = reconstruction.float().clamp(0.0, 1.0)
    prediction = reconstruction * 2.0 - 1.0
    target = target.float() * 2.0 - 1.0
    return perceptual_model(prediction, target).mean()


def train_one_epoch(
    model: TiTok,
    discriminator: PatchDiscriminator,
    perceptual_model: lpips.LPIPS,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    model_scaler: torch.cuda.amp.GradScaler,
    discriminator_scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    config: TrainConfig,
    epoch: int,
    amp_enabled: bool,
) -> dict[str, float]:
    model.train()

    gan_enabled = epoch >= config.gan_start_epoch

    if gan_enabled:
        discriminator.train()
    else:
        discriminator.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_perceptual = 0.0
    total_gan = 0.0
    total_vq = 0.0
    total_discriminator = 0.0
    total_samples = 0

    code_counts = torch.zeros(
        model.config.codebook_size,
        dtype=torch.long,
    )

    for images in loader:
        images = images.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        discriminator_optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            reconstruction, indices, vq_loss = model(images)
            recon_loss = F.mse_loss(reconstruction, images)

            if gan_enabled:
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)

                real_logits = discriminator(images)
                fake_logits = discriminator(
                    reconstruction.detach()
                )

                real_loss = F.binary_cross_entropy_with_logits(
                    real_logits,
                    torch.ones_like(real_logits),
                )
                fake_loss = F.binary_cross_entropy_with_logits(
                    fake_logits,
                    torch.zeros_like(fake_logits),
                )
                discriminator_loss = 0.5 * (
                    real_loss + fake_loss
                )
            else:
                discriminator_loss = torch.zeros(
                    (),
                    device=device,
                )

        if gan_enabled:
            discriminator_scaler.scale(
                discriminator_loss
            ).backward()
            discriminator_scaler.unscale_(
                discriminator_optimizer
            )
            torch.nn.utils.clip_grad_norm_(
                discriminator.parameters(),
                max_norm=config.grad_clip_norm,
            )
            discriminator_scaler.step(discriminator_optimizer)
            discriminator_scaler.update()

            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)

            # Do not update BatchNorm running stats during G step.
            discriminator.eval()

        perceptual_loss = compute_perceptual_loss(
            perceptual_model,
            reconstruction,
            images,
        )

        if gan_enabled:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                fake_logits_for_generator = discriminator(
                    reconstruction
                )
                gan_loss = F.binary_cross_entropy_with_logits(
                    fake_logits_for_generator,
                    torch.ones_like(fake_logits_for_generator),
                )
        else:
            gan_loss = torch.zeros((), device=device)

        generator_loss = (
            config.recon_weight * recon_loss
            + config.perceptual_weight * perceptual_loss
            + config.gan_weight * gan_loss
            + config.vq_loss_weight * vq_loss
        )

        model_scaler.scale(generator_loss).backward()
        model_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.grad_clip_norm,
        )
        model_scaler.step(optimizer)
        model_scaler.update()

        if gan_enabled:
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)
            discriminator.train()

        code_counts += torch.bincount(
            indices.detach().flatten().cpu(),
            minlength=model.config.codebook_size,
        )

        batch_size = images.shape[0]
        total_samples += batch_size
        total_loss += generator_loss.item() * batch_size
        total_recon += recon_loss.item() * batch_size
        total_perceptual += perceptual_loss.item() * batch_size
        total_gan += gan_loss.item() * batch_size
        total_vq += vq_loss.item() * batch_size
        total_discriminator += discriminator_loss.item() * batch_size

    metrics = {
        "loss": total_loss / total_samples,
        "recon": total_recon / total_samples,
        "perceptual": total_perceptual / total_samples,
        "gan": total_gan / total_samples,
        "vq": total_vq / total_samples,
        "discriminator": total_discriminator / total_samples,
    }
    metrics.update(compute_codebook_metrics(code_counts))
    return metrics


@torch.no_grad()
def evaluate(
    model: TiTok,
    perceptual_model: lpips.LPIPS,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: TrainConfig,
    amp_enabled: bool,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_perceptual = 0.0
    total_vq = 0.0
    total_samples = 0
    squared_error = 0.0
    total_values = 0

    code_counts = torch.zeros(
        model.config.codebook_size,
        dtype=torch.long,
    )

    for images in loader:
        images = images.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            reconstruction, indices, vq_loss = model(images)
            recon_loss = F.mse_loss(reconstruction, images)

        perceptual_loss = compute_perceptual_loss(
            perceptual_model,
            reconstruction,
            images,
        )

        loss = (
            config.recon_weight * recon_loss
            + config.perceptual_weight * perceptual_loss
            + config.vq_loss_weight * vq_loss
        )

        batch_size = images.shape[0]
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_recon += recon_loss.item() * batch_size
        total_perceptual += perceptual_loss.item() * batch_size
        total_vq += vq_loss.item() * batch_size

        reconstruction = reconstruction.clamp(0.0, 1.0)
        squared_error += F.mse_loss(
            reconstruction,
            images,
            reduction="sum",
        ).item()
        total_values += images.numel()

        code_counts += torch.bincount(
            indices.detach().flatten().cpu(),
            minlength=model.config.codebook_size,
        )

    mse = squared_error / total_values
    psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))

    metrics = {
        "loss": total_loss / total_samples,
        "recon": total_recon / total_samples,
        "perceptual": total_perceptual / total_samples,
        "vq": total_vq / total_samples,
        "psnr": psnr,
    }
    metrics.update(compute_codebook_metrics(code_counts))
    return metrics


def save_checkpoint(
    path: Path,
    model: TiTok,
    discriminator: PatchDiscriminator,
    model_config: TiTokConfig,
    epoch: int,
    val_loss: float,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "discriminator": discriminator.state_dict(),
            "config": asdict(model_config),
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path,
    )


def main() -> None:
    config = TRAIN_CONFIG

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    if config.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(config.device)

    model_config = TiTokConfig()
    model_config.validate()

    train_loader, val_loader, info = build_dataloaders(
        data_dir=config.data_dir,
        image_size=model_config.image_size,
        batch_size=config.batch_size,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    model = TiTok(model_config).to(device)
    discriminator = PatchDiscriminator(
        in_channels=model_config.in_channels,
    ).to(device)

    perceptual_model = lpips.LPIPS(net="vgg").to(device).eval()
    for parameter in perceptual_model.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.discriminator_lr,
        betas=(0.5, 0.999),
    )

    amp_enabled = config.use_amp and device.type == "cuda"
    model_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )
    discriminator_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)

    generator_parameters = sum(
        p.numel() for p in model.parameters()
    )
    discriminator_parameters = sum(
        p.numel() for p in discriminator.parameters()
    )

    print(f"device        : {device}")
    print(f"total images  : {info['total_size']}")
    print(f"train images  : {info['train_size']}")
    print(f"val images    : {info['val_size']}")
    print(f"model config  : {model_config}")
    print(f"generator p   : {generator_parameters:,}")
    print(f"discriminator : {discriminator_parameters:,}")
    print(f"train batches : {len(train_loader)}")
    print(f"val batches   : {len(val_loader)}")
    print(f"AMP           : {amp_enabled}")
    print(f"GAN start     : epoch {config.gan_start_epoch}")

    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            discriminator=discriminator,
            perceptual_model=perceptual_model,
            loader=train_loader,
            optimizer=optimizer,
            discriminator_optimizer=discriminator_optimizer,
            model_scaler=model_scaler,
            discriminator_scaler=discriminator_scaler,
            device=device,
            config=config,
            epoch=epoch,
            amp_enabled=amp_enabled,
        )
        val_metrics = evaluate(
            model=model,
            perceptual_model=perceptual_model,
            loader=val_loader,
            device=device,
            config=config,
            amp_enabled=amp_enabled,
        )

        save_checkpoint(
            config.output_dir / "last.pt",
            model,
            discriminator,
            model_config,
            epoch,
            val_metrics["loss"],
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                config.output_dir / "best.pt",
                model,
                discriminator,
                model_config,
                epoch,
                val_metrics["loss"],
            )

        if (
            epoch == 1
            or epoch % config.log_every == 0
            or epoch == config.epochs
        ):
            print(
                f"epoch {epoch:03d} | "
                f"train {train_metrics['loss']:.5f} | "
                f"val {val_metrics['loss']:.5f} | "
                f"psnr {val_metrics['psnr']:.2f} dB"
            )
            print(
                f"  recon {train_metrics['recon']:.5f} | "
                f"LPIPS {train_metrics['perceptual']:.5f} | "
                f"GAN {train_metrics['gan']:.5f} | "
                f"vq {train_metrics['vq']:.5f} | "
                f"disc_loss {train_metrics['discriminator']:.5f}"
            )
            print(
                f"  train codes "
                f"{int(train_metrics['unique_codes'])}/"
                f"{model_config.codebook_size} | "
                f"ppl {train_metrics['perplexity']:.2f} | "
                f"dead {train_metrics['dead_code_ratio']:.2%}"
            )
            print(
                f"  val codes   "
                f"{int(val_metrics['unique_codes'])}/"
                f"{model_config.codebook_size} | "
                f"ppl {val_metrics['perplexity']:.2f} | "
                f"dead {val_metrics['dead_code_ratio']:.2%}"
            )

    print(f"best val loss: {best_val_loss:.6f}")
    print(f"checkpoints  : {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()
