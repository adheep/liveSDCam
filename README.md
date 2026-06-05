# LiveSDCam

Real-time webcam transformation with Stable Diffusion. Point your webcam at
yourself, type a prompt, and watch a live transformed feed alongside the source.

Built and tested on **Windows 11 + NVIDIA RTX 5070 Ti (16 GB, Blackwell / sm_120)**.

| | |
|---|---|
| Left | raw webcam |
| Right | live transformed feed |
| Below | prompt, presets, and per-mode settings |

## Modes

| Mode | Technique | Changes | Keeps |
|---|---|---|---|
| **Lite** | SD-Turbo img2img | whole frame (fast, ~7 fps) | nothing locked |
| **Advanced** | + Canny ControlNet + temporal smoothing | whole frame, structure-locked | your structure / pose |
| **Wardrobe** | SD1.5 LCM-inpaint of the clothing region only | **only the outfit** | **face, pose, background** |

- **Lite / Advanced** support an optional **Keep background** toggle (person is
  segmented with DeepLabV3-MobileNet; only the person is transformed).
- **Wardrobe** parses the garment region (`segformer_b2_clothes`), inpaints just
  that area from a prompt, and offers a **📸 Capture (HQ)** button for a clean,
  higher-step one-shot render.

## Requirements

- NVIDIA GPU with recent drivers. For Blackwell (RTX 50-series) you need
  **CUDA 12.8** wheels (PyTorch 2.7+; this repo uses 2.11).
- Python 3.12 recommended.
- A connected webcam.

## Setup

```bash
# 1. create a virtual environment (Python 3.12)
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/Mac

# 2. install PyTorch from the CUDA 12.8 index (Blackwell support)
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# 3. install the rest
pip install -r requirements.txt
```

Verify the GPU is seen:

```bash
python check_gpu.py
```

## Run

```bash
python app.py
```

Opens at **http://127.0.0.1:7860**. Click the **⏺ Record** button on the webcam
to start streaming. Models download from Hugging Face on first use
(SD-Turbo ~2.5 GB; the ControlNet / inpaint / parser models load lazily the
first time you switch to that mode).

## How it works

Each engine keeps the diffusion pipeline warm and uses a **single-flight lock**:
if the GPU is still busy with the previous frame, the new frame is dropped (the
last output is shown) so the feed stays live with no lag buildup. A frame-rate
read-out is shown under the transformed feed.

| File | Role |
|---|---|
| `app.py` | Gradio UI, mode switching, lazy engine loading |
| `engine.py` | Lite engine (SD-Turbo img2img) |
| `engine_advanced.py` | Advanced engine (Canny ControlNet + temporal smoothing) |
| `engine_wardrobe.py` | Wardrobe engine (clothes parser + SD1.5 LCM-inpaint, live + capture) |
| `segmenter.py` | Person segmenter + background-composite helper (Keep background) |
| `check_gpu.py` | GPU / CUDA sanity check |
| `smoke_*.py`, `test_*.py` | Standalone validation scripts used during development |

## Notes

- Frame-to-frame flicker is inherent to per-frame diffusion; **Stabilize**
  (fixed seed) and **Temporal smoothing** reduce it, and ControlNet/inpaint
  modes are steadier because they anchor to your real structure.
- The whole pipeline is currently **unoptimized** (no `torch.compile` / TensorRT /
  TAESD) — there is significant headroom for higher framerates.

## Models used

- [`stabilityai/sd-turbo`](https://huggingface.co/stabilityai/sd-turbo)
- [`thibaud/controlnet-sd21-canny-diffusers`](https://huggingface.co/thibaud/controlnet-sd21-canny-diffusers)
- [`Lykon/dreamshaper-8-inpainting`](https://huggingface.co/Lykon/dreamshaper-8-inpainting) + [`latent-consistency/lcm-lora-sdv1-5`](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5)
- [`mattmdjaga/segformer_b2_clothes`](https://huggingface.co/mattmdjaga/segformer_b2_clothes)
- torchvision DeepLabV3-MobileNet (person segmentation)
