import torch
from torch import nn
from titok.config import TiTokConfig

class PatchEmbed(nn.Module):
    """把图像切分成 patch token。"""

    def __init__(self, config: TiTokConfig) -> None: #-> None返回值类型注解，表示这个函数返回 None
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.hidden_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, config.num_patches, config.hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        '''可学习的位置编码（learnable positional embedding）先创建全 0 参数，再用截断正态随机填充；训练时这些值会被优化器不断更新。
        原地初始化为均值 0、标准差 0.02、范围约 [-0.04, 0.04] 的截断正态随机数'''

    def forward(self, images:torch.tensor) -> torch.Tensor:
        x = self.proj(images)
        x = x.flatten(2).transpose(1,2)
        return x + self.pos_embed     # (B, N, D) + (1, N, D) 广播 N = H' * W'


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
#Pre-LN（这段代码）,先归一再残差。原始论文是先残差，再归一。这种最终输出没有经过 norm，通常需要在所有 block 之后再加一个 final_norm。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(h)

        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class LatentEncoder(nn.Module):
    """把图像压缩成固定数量的 latent token。"""

    def __init__(self, config: TiTokConfig) -> None:
        super().__init__()

        self.patch_embed = PatchEmbed(config)
        self.latent_tokens = nn.Parameter(
            torch.zeros(1, config.latent_tokens, config.hidden_dim)
        )
        self.latent_pos_embed = nn.Parameter(
            torch.zeros(1, config.latent_tokens, config.hidden_dim)
        )
        nn.init.trunc_normal_(self.latent_tokens, std=0.02)
        nn.init.trunc_normal_(self.latent_pos_embed, std=0.02)

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

        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(images)

        latents = self.latent_tokens + self.latent_pos_embed
        latents = latents.expand(images.shape[0], -1, -1)

        x = torch.cat([latents, patches], dim=1)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, : self.latent_tokens.shape[1]]