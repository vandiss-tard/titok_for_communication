from dataclasses import dataclass


@dataclass(slots=True)
class TiTokConfig:
    """TiTok 模型参数。"""

    # 输入图像
    image_size: int = 256
    patch_size: int = 16
    in_channels: int = 3

    # 1D latent token
    latent_tokens: int = 32
    token_size: int = 12

    # Transformer
    hidden_dim: int = 512
    depth: int = 10
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    # VQ 码本
    codebook_size: int = 4096
    commitment_weight: float = 0.25
    ema_decay: float = 0.99
    ema_eps: float = 1e-5

    # 死码重启
    dead_code_threshold: int = 100
    dead_code_restart_interval: int = 100
    max_restart_codes: int = 32

    @property
    def num_patches(self) -> int:
        patches_per_side = self.image_size // self.patch_size
        return patches_per_side * patches_per_side

    def validate(self) -> None:
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        if self.token_size <= 0:
            raise ValueError("token_size must be positive")

        if self.dead_code_restart_interval <= 0:
            raise ValueError(
                "dead_code_restart_interval must be positive"
            )

        if self.max_restart_codes <= 0:
            raise ValueError("max_restart_codes must be positive")