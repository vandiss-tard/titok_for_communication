'''
负责把神经网络输出转换成真正可传输的二进制数据
bits_per_token(4096) -> 12  计算每个 token 需要多少 bit。
发送端使用 indices_to_bytes[B, 32] 的整数 -> [B, 48] 的 uint8 字节.
接收端使用 bytes_to_indices  [B, 48] 的字节 -> [B, 32] 的整数
'''


import math

import torch


def bits_per_token(codebook_size: int) -> int:
    """返回每个 token 所需的 bit 数。"""
    if codebook_size < 2:
        raise ValueError("codebook_size must be at least 2")

    return math.ceil(math.log2(codebook_size))


def indices_to_bytes(
    indices: torch.Tensor,
    codebook_size: int,
) -> torch.Tensor:
    """把 [B, N] 的 token 编号打包成 [B, bytes] 字节。"""
    if indices.ndim != 2:
        raise ValueError("indices must have shape [B, N]")

    width = bits_per_token(codebook_size)
    indices = indices.to(torch.int64)

    if indices.numel() > 0:
        if indices.min().item() < 0:
            raise ValueError("indices must be non-negative")
        if indices.max().item() >= codebook_size:
            raise ValueError("indices exceed codebook size")

    # 每个编号展开成 width 个 bit。
    shifts = torch.arange(
        width - 1,
        -1,
        -1,
        device=indices.device,
    )
    bits = ((indices.unsqueeze(-1) >> shifts) & 1).to(torch.uint8)
    bits = bits.reshape(indices.shape[0], -1).cpu()

    # 补齐到 8 bit 的整数倍。
    padding = (-bits.shape[1]) % 8
    if padding > 0:
        zeros = torch.zeros(
            bits.shape[0],
            padding,
            dtype=torch.uint8,
        )
        bits = torch.cat([bits, zeros], dim=1)

    # 每 8 bit 合并成一个字节。
    groups = bits.reshape(bits.shape[0], -1, 8)
    weights = torch.tensor(
        [128, 64, 32, 16, 8, 4, 2, 1],
        dtype=torch.int64,
    )
    packed = (groups.long() * weights).sum(dim=-1)
    return packed.to(torch.uint8)


def bytes_to_indices(
    data: torch.Tensor,
    num_tokens: int,
    codebook_size: int,
) -> torch.Tensor:
    """把 [B, bytes] 字节还原成 [B, N] token 编号。"""
    if data.ndim != 2:
        raise ValueError("data must have shape [B, bytes]")
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")

    width = bits_per_token(codebook_size)
    total_bits = num_tokens * width

    data = data.cpu().to(torch.uint8)
    if data.shape[1] * 8 < total_bits:
        raise ValueError("Not enough bytes to decode all tokens")

    # 每个字节拆成 8 个 bit，顺序与打包时一致。
    shifts = torch.arange(7, -1, -1)
    bits = ((data.unsqueeze(-1) >> shifts) & 1).to(torch.uint8)

    bits = bits.reshape(data.shape[0], -1)
    bits = bits[:, :total_bits]
    bits = bits.reshape(data.shape[0], num_tokens, width)

    # 把每个 token 的 bit 重新组合成整数。
    token_shifts = torch.arange(width - 1, -1, -1)
    return (bits.long() << token_shifts).sum(dim=-1)