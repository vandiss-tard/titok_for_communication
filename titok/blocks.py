from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from titok.config import TiTokConfig


class PatchEmbed(nn.Module):
    """把图像切分成 patch token。"""

    def __init__(self, config: TiTokConfig) -> None:
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.hidden_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches, config.hidden_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.proj(images)
        x = x.flatten(2).transpose(1, 2)
        return x + self.pos_embed


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()

        hidden_dim = int(dim * mlp_ratio)

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(h)

        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class LatentEncoder(nn.Module):
    """把图像编码成固定数量的 1D latent token。"""

    def __init__(self, config: TiTokConfig) -> None:
        super().__init__()

        self.num_patches = config.num_patches
        self.num_latent_tokens = config.latent_tokens

        self.patch_embed = PatchEmbed(config)

        self.class_token = nn.Parameter(
            torch.zeros(1, 1, config.hidden_dim)
        )
        self.latent_tokens = nn.Parameter(
            torch.zeros(1, config.latent_tokens, config.hidden_dim)
        )
        self.latent_pos_embed = nn.Parameter(
            torch.zeros(1, config.latent_tokens, config.hidden_dim)
        )

        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.latent_tokens, std=0.02)
        nn.init.trunc_normal_(self.latent_pos_embed, std=0.02)

        self.ln_pre = nn.LayerNorm(config.hidden_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.depth)
            ]
        )

        self.ln_post = nn.LayerNorm(config.hidden_dim)
        self.to_token_size = nn.Linear(
            config.hidden_dim,
            config.token_size,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]

        patches = self.patch_embed(images)
        class_token = self.class_token.expand(
            batch_size,
            -1,
            -1,
        )
        latents = (
            self.latent_tokens + self.latent_pos_embed
        ).expand(batch_size, -1, -1)

        x = torch.cat(
            [class_token, patches, latents],
            dim=1,
        )
        x = self.ln_pre(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)

        latent_start = 1 + self.num_patches
        latent_end = latent_start + self.num_latent_tokens
        latents = x[:, latent_start:latent_end]

        return self.to_token_size(latents)


class VectorQuantizer(nn.Module):
    """L2 归一化、EMA 更新和死码重启的向量量化器。"""

    def __init__(self, config: TiTokConfig) -> None:
        super().__init__()

        self.codebook_size = config.codebook_size
        self.token_size = config.token_size
        self.commitment_weight = config.commitment_weight
        self.ema_decay = config.ema_decay
        self.ema_eps = config.ema_eps

        self.dead_code_threshold = config.dead_code_threshold
        self.dead_code_restart_interval = (
            config.dead_code_restart_interval
        )
        self.max_restart_codes = config.max_restart_codes

        codebook = torch.randn(
            config.codebook_size,
            config.token_size,
        )
        codebook = F.normalize(codebook, dim=-1)

        self.register_buffer("codebook", codebook)
        self.register_buffer(
            "cluster_size",
            torch.ones(config.codebook_size),
        )
        self.register_buffer(
            "codebook_sum",
            codebook.clone(),
        )
        self.register_buffer(
            "steps_since_used",
            torch.zeros(
                config.codebook_size,
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "restart_step",
            torch.zeros((), dtype=torch.long),
        )

    @torch.no_grad()
    def _update_code_usage(
        self,
        indices: torch.Tensor,
    ) -> None:
        used = torch.zeros(
            self.codebook_size,
            dtype=torch.bool,
            device=indices.device,
        )
        used[indices] = True

        self.steps_since_used[used] = 0
        self.steps_since_used[~used] += 1

    @torch.no_grad()
    def _ema_update(
        self,
        z_norm: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        one_hot = F.one_hot(
            indices,
            self.codebook_size,
        ).to(z_norm.dtype)

        batch_cluster_size = one_hot.sum(dim=0)
        batch_codebook_sum = one_hot.transpose(0, 1) @ z_norm

        decay = self.ema_decay
        update = 1.0 - decay

        self.cluster_size.mul_(decay).add_(
            batch_cluster_size,
            alpha=update,
        )
        self.codebook_sum.mul_(decay).add_(
            batch_codebook_sum,
            alpha=update,
        )

        total = self.cluster_size.sum()
        cluster_size = (
            (self.cluster_size + self.ema_eps)
            / (
                total
                + self.codebook_size * self.ema_eps
            )
            * total
        )

        self.codebook.copy_(
            self.codebook_sum / cluster_size.unsqueeze(1)
        )
        self.codebook.copy_(
            F.normalize(self.codebook, dim=-1)
        )

    @torch.no_grad()
    def _restart_dead_codes(
        self,
        z_norm: torch.Tensor,
    ) -> None:
        self.restart_step += 1

        if (
            self.restart_step.item()
            % self.dead_code_restart_interval
            != 0
        ):
            return

        dead_codes = torch.nonzero(
            self.steps_since_used
            >= self.dead_code_threshold,
            as_tuple=False,
        ).flatten()

        if dead_codes.numel() == 0:
            return

        restart_count = min(
            dead_codes.numel(),
            self.max_restart_codes,
        )

        order = torch.randperm(
            dead_codes.numel(),
            device=dead_codes.device,
        )
        restart_codes = dead_codes[order[:restart_count]]

        random_positions = torch.randint(
            0,
            z_norm.shape[0],
            (restart_count,),
            device=z_norm.device,
        )
        replacement = z_norm[random_positions].detach()

        self.codebook[restart_codes] = replacement
        self.codebook_sum[restart_codes] = replacement
        self.cluster_size[restart_codes] = 1.0
        self.steps_since_used[restart_codes] = 0

    def forward(
        self,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, dim = z.shape
        z_flat = z.reshape(-1, dim).float()

        z_norm = F.normalize(z_flat, dim=-1)
        codebook_norm = F.normalize(
            self.codebook,
            dim=-1,
        )

        distances = torch.cdist(
            z_norm,
            codebook_norm,
        )
        indices = distances.argmin(dim=1)

        quantized_norm = codebook_norm[indices]

        if self.training:
            self._update_code_usage(indices)
            self._ema_update(z_norm, indices)
            self._restart_dead_codes(z_norm)

        commitment_loss = F.mse_loss(
            z_norm,
            quantized_norm.detach(),
        )
        vq_loss = self.commitment_weight * commitment_loss

        quantized = z_norm + (
            quantized_norm - z_norm
        ).detach()
        quantized = quantized.view(
            batch_size,
            num_tokens,
            dim,
        )

        return (
            quantized,
            indices.view(batch_size, num_tokens),
            vq_loss,
        )

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        quantized = self.codebook[indices].float()
        return F.normalize(quantized, dim=-1)


class LatentDecoder(nn.Module):
    """使用联合自注意力把 latent token 重建为图像。"""

    def __init__(self, config: TiTokConfig) -> None:
        super().__init__()

        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.num_patches = config.num_patches
        self.patches_per_side = config.image_size // config.patch_size

        self.decoder_embed = nn.Linear(
            config.token_size,
            config.hidden_dim,
        )

        self.class_token = nn.Parameter(
            torch.zeros(1, 1, config.hidden_dim)
        )
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, config.hidden_dim)
        )
        self.mask_pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches, config.hidden_dim)
        )
        self.latent_pos_embed = nn.Parameter(
            torch.zeros(
                1,
                config.latent_tokens,
                config.hidden_dim,
            )
        )

        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.mask_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.latent_pos_embed, std=0.02)

        self.ln_pre = nn.LayerNorm(config.hidden_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.depth)
            ]
        )

        self.ln_post = nn.LayerNorm(config.hidden_dim)

        patch_dim = (
            config.patch_size
            * config.patch_size
            * config.in_channels
        )
        self.head = nn.Linear(
            config.hidden_dim,
            patch_dim,
        )
        self.output_conv = nn.Conv2d(
            config.in_channels,
            config.in_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        batch_size = memory.shape[0]

        memory = self.decoder_embed(memory)
        memory = memory + self.latent_pos_embed

        class_token = self.class_token.expand(
            batch_size,
            -1,
            -1,
        )
        mask_tokens = self.mask_token.expand(
            batch_size,
            self.num_patches,
            -1,
        )
        mask_tokens = mask_tokens + self.mask_pos_embed

        x = torch.cat(
            [class_token, mask_tokens, memory],
            dim=1,
        )
        x = self.ln_pre(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)

        mask_output = x[:, 1 : 1 + self.num_patches]
        patches = self.head(mask_output)

        side = self.patches_per_side
        p = self.patch_size
        c = self.in_channels

        image = patches.reshape(
            batch_size,
            side,
            side,
            p,
            p,
            c,
        )
        image = image.permute(0, 5, 1, 3, 2, 4)
        image = image.reshape(
            batch_size,
            c,
            side * p,
            side * p,
        )

        return self.output_conv(image)


class TiTok(nn.Module):
    """完整的图像编码、量化和重建模型。"""

    def __init__(self, config: TiTokConfig) -> None:
        super().__init__()

        self.config = config
        self.encoder = LatentEncoder(config)
        self.quantizer = VectorQuantizer(config)
        self.decoder = LatentDecoder(config)

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latents = self.encoder(images)
        quantized, indices, vq_loss = self.quantizer(latents)
        reconstruction = self.decoder(quantized)

        return reconstruction, indices, vq_loss

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        latents = self.encoder(images)
        _, indices, _ = self.quantizer(latents)
        return indices

    @torch.no_grad()
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        quantized = self.quantizer.lookup(indices)
        return self.decoder(quantized)