from dataclasses import dataclass
'''
# dataclass 用来快速定义“只负责保存参数”的类。
# 使用它后，不需要手写 __init__ 初始化函数。

# @dataclass 会自动为类生成初始化方法，例如：
# config = TiTokConfig(image_size=256)
#
# slots=True 表示这个类只允许使用下面声明过的属性，
# 可以减少内存占用，并防止不小心拼错属性名。
'''
@dataclass(slots=True)
class TiTokConfig:
    """TiTok 模型参数。"""

    # 输入图像：256 × 256 RGB
    image_size: int = 256
    patch_size: int = 16
    in_channels: int = 3

    # 将图像压缩成 32 个一维 token
    latent_tokens: int = 32

    # Transformer 参数
    hidden_dim: int = 384
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0                     # MLP 的隐藏层维度 = hidden_dim * mlp_ratio
    dropout: float = 0.0

    # VQ 码本参数
    codebook_size: int = 4096
    commitment_weight: float = 0.25

    # @property 表示 num_patches 不是一个普通变量，
    # 而是根据 image_size 和 patch_size 自动计算出来的属性。  
    
    @property
    def num_patches(self) -> int:
        """返回图像切分后的 patch 数量。"""
        patches_per_side = self.image_size // self.patch_size
        return patches_per_side * patches_per_side

    def validate(self) -> None:
        """检查参数是否合法。"""
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")