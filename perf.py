"""Shared GPU optimizations for the diffusion pipelines (Blackwell / 5070 Ti).

- TF32 + flash SDP attention backends
- channels_last memory format (faster convs)
- TAESD tiny VAE for fast latent decode (live preview; ~10x faster than full VAE)
- torch.compile on the UNet (kernel fusion)

All wrapped defensively so a failure degrades to the unoptimized pipe, never crashes.
"""
import cv2
import numpy as np
import torch
from diffusers import AutoencoderTiny


def sharpen(arr, amount=0.7, sigma=1.2):
    """Unsharp mask (~1ms) to recover crispness lost by the fast TAESD decoder."""
    if arr is None or amount <= 0:
        return arr
    blur = cv2.GaussianBlur(arr, (0, 0), sigma)
    return np.clip(arr.astype(np.float32) * (1 + amount) - blur.astype(np.float32) * amount,
                   0, 255).astype(np.uint8)

_flags_set = False
_taesd = None


def _global_flags():
    global _flags_set
    if _flags_set:
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except Exception:
        pass
    _flags_set = True


def get_taesd(dtype, device):
    """Shared TAESD instance (works for SD1.x / SD2.x latents)."""
    global _taesd
    if _taesd is None:
        _taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=dtype).to(device)
    return _taesd


def optimize_pipe(pipe, dtype, device, use_taesd=True, compile_unet=True, label=""):
    _global_flags()
    try:
        pipe.unet.to(memory_format=torch.channels_last)
    except Exception as e:
        print(f"[perf{label}] channels_last skipped: {e}")
    if use_taesd:
        try:
            pipe.vae = get_taesd(dtype, device)
            print(f"[perf{label}] TAESD fast VAE enabled")
        except Exception as e:
            print(f"[perf{label}] TAESD skipped: {e}")
    if compile_unet:
        try:
            pipe.unet = torch.compile(pipe.unet, fullgraph=False)
            print(f"[perf{label}] torch.compile(unet) enabled (compiles on first frames)")
        except Exception as e:
            print(f"[perf{label}] torch.compile skipped: {e}")
    return pipe
