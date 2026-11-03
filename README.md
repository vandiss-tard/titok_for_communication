# TiTok Image Compression and Transmission

本项目实现一个面向图像压缩传输的简化 TiTok 风格 1D 图像 tokenizer。
项目使用 32 个离散 latent token 表示一张 `256 x 256` RGB 图像，并提供训练、
二进制索引打包、索引还原和图像重建流程。

本项目只关注图像重建，不包含字节官方 TiTok 的生成式 MaskGIT/MaskGen/RAR 模块。

## 1. Current Features

- `256 x 256` RGB image input
- ViT-style patch embedding
- Class token + patch tokens + 1D latent tokens
- 32 latent tokens per image
- 10-layer Transformer encoder and decoder
- L2-normalized vector quantization
- EMA codebook update
- Dead-code restart
- Joint self-attention decoder
- Single-stage VQGAN training
- MSE + LPIPS + GAN + commitment losses
- PatchGAN discriminator
- Mixed precision training
- Codebook usage diagnostics
- Binary token index packing and unpacking
- Encode/decode entry points

## 2. Project Layout

```text
titok_for_communication/
├── train.py                    # Main training entry point
├── encode.py                   # Image -> VQ indices -> binary file
├── decode.py                   # Binary file -> VQ indices -> reconstructed image
├── scripts/
│   └── dataloader.py           # Dataset, split, augmentation, DataLoader
├── titok/
│   ├── config.py               # All model hyperparameters
│   ├── blocks.py               # Encoder, VQ, decoder, TiTok wrapper
│   ├── discriminator.py        # PatchGAN discriminator
│   └── Bin_transmission.py     # Binary index packing and unpacking
├── data/
│   └── train/                  # Training images
├── runs/
│   └── train/                  # best.pt and last.pt
├── pre_transmit_data/          # Encoded .bin files
└── After_transmit_data/        # Reconstructed images
```

## 3. Environment

Tested local environment:

```text
Python       3.10.20
PyTorch      2.13.0+cu130
torchvision  0.28.0+cu130
Pillow       12.3.0
LPIPS        0.1.4
NumPy        2.2.6
SciPy        1.15.2
CUDA         available
```

Recommended server setup:

```bash
conda create -n titok python=3.10 -y
conda activate titok
```

Install the PyTorch version matching the server CUDA version first. Example for CUDA 12.1:

```bash
pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The first LPIPS run may download the torchvision VGG16 weights. For an offline
server, pre-download VGG16 into `~/.cache/torch/hub/checkpoints/`.

## 4. Image Data

Place training images under:

```text
data/train/
```

Supported formats:

```text
.jpg
.jpeg
.png
```

Training/validation split is deterministic and controlled by `val_ratio`.

Training augmentation:

```text
Resize shorter edge
RandomCrop 256 x 256
RandomHorizontalFlip
ToTensor
```

Validation preprocessing:

```text
Resize shorter edge
CenterCrop 256 x 256
ToTensor
```

## 5. Model Configuration

All model parameters are defined in `titok/config.py`.

Current defaults:

```text
image_size       = 256
patch_size       = 16
in_channels      = 3
latent_tokens    = 32
token_size       = 12
hidden_dim       = 512
depth            = 10
num_heads        = 8
mlp_ratio        = 4.0
dropout          = 0.0
codebook_size    = 4096
commitment_weight = 0.25
ema_decay        = 0.99
ema_eps          = 1e-5
dead_code_threshold = 100
dead_code_restart_interval = 100
max_restart_codes = 32
```

Current generator parameter count:

```text
64,165,216
```

Current discriminator parameter count:

```text
2,766,529
```

## 6. Network Architecture

### 6.1 Patch Embedding

Input:

```text
[B, 3, 256, 256]
```

A `Conv2d` with:

```text
kernel_size = 16
stride      = 16
out_channels = 512
```

produces:

```text
[B, 256, 512]
```

A learnable positional embedding is added to the 256 patch tokens.

### 6.2 Encoder

Learnable class token:

```text
[1, 1, 512]
```

Learnable latent tokens:

```text
[1, 32, 512]
```

Learnable latent positional embeddings:

```text
[1, 32, 512]
```

Sequence construction:

```text
class token + 256 patch tokens + 32 latent tokens
= [B, 289, 512]
```

The sequence passes through:

```text
LayerNorm
10 Transformer blocks
LayerNorm
```

Each Transformer block contains:

```text
Pre-LayerNorm
8-head MultiheadAttention
Residual connection
Pre-LayerNorm
MLP: 512 -> 2048 -> 512
Residual connection
```

The 32 latent positions are extracted and projected:

```text
[B, 32, 512] -> Linear(512, 12) -> [B, 32, 12]
```

### 6.3 Vector Quantizer

Codebook:

```text
[4096, 12]
```

Both encoder latents and codebook vectors are L2 normalized. Nearest codes are
selected using Euclidean distance, which is equivalent to cosine similarity for
unit-length vectors.

Codebook updates use EMA buffers:

```text
codebook
cluster_size
codebook_sum
steps_since_used
restart_step
```

Dead codes are periodically replaced with latents sampled from the current
training batch. The codebook itself is not optimized by a codebook MSE loss;
only the commitment loss updates the encoder.

Output:

```text
quantized latents: [B, 32, 12]
indices:           [B, 32]
```

### 6.4 Decoder

Quantized indices are mapped back to unit-length codebook vectors:

```text
[B, 32] -> [B, 32, 12]
```

Decoder projection:

```text
Linear(12, 512) -> [B, 32, 512]
```

Decoder sequence construction:

```text
class token + 256 mask tokens + 32 latent tokens
= [B, 289, 512]
```

The mask tokens receive positional embeddings. The latent tokens receive their
own positional embeddings. The full sequence passes through:

```text
LayerNorm
10 joint self-attention Transformer blocks
LayerNorm
```

The 256 mask outputs are extracted and projected:

```text
[B, 256, 512] -> Linear(512, 768)
```

The 768 values are reshaped into:

```text
16 x 16 patches x 16 x 16 pixels x 3 channels
```

Finally, a `3 x 3 Conv2d` smooths patch boundaries and produces:

```text
[B, 3, 256, 256]
```

### 6.5 Discriminator

PatchGAN discriminator:

```text
Conv 3 -> 64
Conv 64 -> 128 + BatchNorm + LeakyReLU
Conv 128 -> 256 + BatchNorm + LeakyReLU
Conv 256 -> 512 + BatchNorm + LeakyReLU
Conv 512 -> 1
```

It returns spatial patch logits.

## 7. Training Objective

Generator loss:

```text
generator_loss =
    recon_weight       * MSE(reconstruction, target)
  + perceptual_weight  * LPIPS(reconstruction, target)
  + gan_weight         * GAN_generator_loss
  + vq_loss_weight     * commitment_loss
```

Current weights:

```text
recon_weight       = 1.0
perceptual_weight  = 0.1
gan_weight         = 0.01
vq_loss_weight     = 1.0
commitment_weight  = 0.25
```

Discriminator loss:

```text
discriminator_loss =
0.5 * [
    BCE(D(real), 1)
  + BCE(D(fake), 0)
]
```

Training uses:

```text
AdamW for generator
Adam for discriminator
AMP fp16 on CUDA
gradient clipping
best.pt / last.pt checkpoints
```

## 8. Training

Run from the project root:

```bash
cd /path/to/titok_for_communication
python train.py
```

Default training configuration is defined at the top of `train.py`:

```text
batch_size = 1
val_ratio = 0.2
epochs = 20
lr = 3e-4
discriminator_lr = 1e-4
gan_start_epoch = 1
use_amp = True
```

Output checkpoints:

```text
runs/train/best.pt
runs/train/last.pt
```

Each checkpoint contains:

```text
model state_dict
discriminator state_dict
model config
epoch
validation loss
```

Training log fields:

```text
train / val  generator total loss
psnr         validation PSNR
recon        reconstruction MSE
perceptual   LPIPS
gan          generator adversarial loss
vq           commitment loss
disc         discriminator loss
train codes  unique code count
ppl          code distribution perplexity
dead         unused code ratio for the current epoch
```

The current dataset is very small. For meaningful image quality, use a much
larger and more diverse dataset.

## 9. Binary Token Transmission

For the current configuration:

```text
32 tokens
codebook size = 4096 = 2^12
bits per token = 12
total bits = 32 x 12 = 384 bit
total bytes = 48 Byte
```

The `.bin` file contains raw packed token indices. It does not contain image
pixels or model weights.

### Encode

Edit the path constants at the top of `encode.py`, then run:

```bash
python encode.py
```

Output:

```text
pre_transmit_data/*_indices.bin
```

### Decode

Edit the path constants at the top of `decode.py`, then run:

```bash
python decode.py
```

Output:

```text
After_transmit_data/*_reconstruction.png
```

The encoder and decoder must use the same checkpoint and codebook configuration.

## 10. Server Notes

### Server run checklist

Check the GPU and PyTorch CUDA state:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Run a long training job inside `tmux`:

```bash
tmux new -s titok
cd /path/to/titok_for_communication
python train.py 2>&1 | tee train.log
```

Detach from `tmux` with `Ctrl+B`, then `D`. Reattach with:

```bash
tmux attach -t titok
```

Before a full run, verify imports and model construction:

```bash
python -m py_compile train.py encode.py decode.py scripts/dataloader.py   titok/config.py titok/blocks.py titok/discriminator.py titok/Bin_transmission.py
```

- Run commands from the project root.
- `scripts/dataloader.py` currently uses `num_workers=0`; on a server with more CPU
  workers, this can be increased.
- The code defaults to `batch_size=1` to fit a 6 GB GPU. Larger GPUs can increase
  batch size.
- `runs/`, `data/`, and model weights are intentionally excluded from Git.
- Validate a new checkpoint with both `encode.py` and `decode.py` before using it
  for transmission experiments.
- The current single-stage VQGAN training is not an exact reproduction of the
  official two-stage TiTok training pipeline.
- The binary transport currently contains only packed indices. CRC, packet
  headers, forward error correction, and channel simulation are future work.
