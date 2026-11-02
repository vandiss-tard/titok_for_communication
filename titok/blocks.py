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

    def forward(self, images:torch.tensor) -> torch.tensor:
        x = self.proj(images)
        x = x.flatten(2).transpose(1,2)
        return x + self.pos_embed     # (B, N, D) + (1, N, D) 广播 N = H' * W'
    