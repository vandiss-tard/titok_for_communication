import torch
from torch import nn
from titok.config import TiTokConfig
from torch.nn import functional as F


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
        latents = latents.expand(images.shape[0], -1, -1) # batch 维从 1 扩展到当前 batch 大小，让每个样本都能拿到一组 latent token。-1表示该维度不变

        x = torch.cat([latents, patches], dim=1)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        '''等价于
        x = self.blocks[0](x)
        x = self.blocks[1](x)
        x = self.blocks[2](x)
        ...
        x = self.blocks[depth - 1](x)
        '''
        return x[:, : self.latent_tokens.shape[1]]
        '''
        x.shape == (B, L + N, D)
        :：第 0 维（batch）全取，B 个样本都保留
        : self.latent_tokens.shape[1]：第 1 维取 [0, L)，即前 L 个 token
        '''


class VectorQuantizer(nn.Module):
    """把连续 latent 映射到 VQ 码本编号。"""

    def __init__(self, config: TiTokConfig) -> None:
        super().__init__()

        self.commitment_weight = config.commitment_weight
        self.codebook = nn.Embedding( 
            config.codebook_size,
            config.hidden_dim,
        )
        self.codebook.weight.data.uniform_( ##初始化为 [-1/K, 1/K] 的均匀分布
            -1.0 / config.codebook_size,
            1.0 / config.codebook_size,
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, dim = z.shape
        z_flat = z.reshape(-1, dim)  #新形状为(-1, D)，-1让pytorch自己算

#torch.cdist(A, B) 计算 A 最后一维中每一行到 B最后一维 中每一行的欧氏距离
        distances = torch.cdist(
            z_flat.float(),
            self.codebook.weight.float(),
        )
        indices = distances.argmin(dim=1)#沿着第 1 维（K （码本向量数）这一维）去找最小值，并返回它在第 1 维上的位置
        quantized = self.codebook(indices) #n.Embedding 的前向就是按索引取行
        quantized = quantized.view(batch_size, num_tokens, dim)

        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(z, quantized.detach())
        vq_loss = (
            codebook_loss
            + self.commitment_weight * commitment_loss
        )

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        return (
            quantized,
            indices.view(batch_size, num_tokens),
            vq_loss,
        )

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        return self.codebook(indices)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm3 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, need_weights=False) #nn.MutiAttention在forward中返回(attn_output, attn_weights)
        x = x + self.dropout(h)

        h, _ = self.cross_attn(
            self.norm2(x),
            memory,
            memory,
            need_weights=False,
        )
        x = x + self.dropout(h)

        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x
        '''
        attn_output, attn_weights = self.attn(
    query,                    # (B, N_q, D)
    key,                      # (B, N_k, D)
    value,                    # (B, N_k, D)
    key_padding_mask=None,    # (B, N_k)，True 表示该 key 是 padding，忽略
    need_weights=True,        # 是否返回注意力权重
    attn_mask=None,           # (N_q, N_k) 或 (B*num_heads, N_q, N_k)
    average_attn_weights=True,# 多头权重是否平均
    is_causal=False,          # True 时自动生成因果掩码
)       
        '''