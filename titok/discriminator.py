from __future__ import annotations

import torch
from torch import nn


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for single-stage VQGAN training."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
    ) -> None:
        super().__init__()

        def block(
            in_ch: int,
            out_ch: int,
            normalize: bool,
        ) -> list[nn.Module]:
            layers: list[nn.Module] = [
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(in_channels, base_channels, False),
            *block(base_channels, base_channels * 2, True),
            *block(base_channels * 2, base_channels * 4, True),
            *block(base_channels * 4, base_channels * 8, True),
            nn.Conv2d(
                base_channels * 8,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)
