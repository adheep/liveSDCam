# LiveSDCam

Real-time webcam transformation with Stable Diffusion. Point a camera at yourself,
type a prompt (or attach a garment), and watch a live transformed feed.

Built and tested on **Windows 11 + NVIDIA RTX 5070 Ti (16 GB, Blackwell / sm_120)**.

Two front-ends, same engines:
- **`/`** — Gradio UI: side-by-side source + transformed feeds, full controls (great on desktop).
- **`/live`** — custom **FaceTime-style** page: full-screen transformed result + a
  picture-in-picture camera, collapsible controls, front/back camera toggle. Ideal on a phone.

## Modes

| Mode | Technique | Changes | Keeps |
|---|---|---|---|
| **Lite** | SD-Turbo img2img | whole frame (fast) | nothing locked |
| **Wardrobe** | SD1.5 **LCM-inpaint** of the clothing region (from a text prompt) | **only the outfit** | **face, pose, background** |
| **Try-On** | attach a **garment image** → IP-Adapter (live) + CatVTON (HQ capture) | **only the outfit**, matched to your photo | face, pose, background |

- **Lite** supports an optional **Keep background** toggle (person segmented with
  DeepLabV3-MobileNet; only the person is transformed).
- **Wardrobe / Try-On** parse the garment region (`segformer_b2_clothes`) and inpaint
  just that area, with a **📸 Capture (HQ)** button for a clean, higher-quality render
  (Wardrobe = full-VAE inpaint; Try-On = CatVTON faithful try-on of the exact garment).
- **Try-On** auto-captions the uploaded garment (BLIP) so the prompt matches the image.

## Requirements

- NVIDIA GPU + recent drivers. For Blackwell (RTX 50-series): **CUDA 12.8** wheels (PyTorch 2.7+; this repo uses 2.11).
- Python 3.12 recommended.
- A connected webcam (or a phone via `/live`).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows  (Linux/Mac: source .venv/bin/activate)

# PyTorch from the CUDA 12.8 index (Blackwell support)
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt        # includes triton-windows for torch.compile (Windows)

python check_gpu.py                     # sanity-check the GPU
python download_models.py               # (recommended) pre-fetch models (~6 GB, cached, not in repo)
```

**Try-On HQ Capture only** also needs the CatVTON repo (cloned locally, not vendored here;
note its **non-commercial** license):

```bash
git clone https://github.com/Zheng-Chong/CatVTON third_party/CatVTON
```

## Run

```bash
python app.py
```

- Desktop UI: **http://127.0.0.1:7860/**
- Immersive page: **http://127.0.0.1:7860/live**

If a self-signed cert (`cert.pem` / `key.pem`) is present, the app serves **HTTPS on the LAN**
(`0.0.0.0`) so a phone on the same Wi-Fi can use its camera (browsers require HTTPS off
localhost). Generate one with:

```bash
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 825 -nodes \
  -subj "/CN=LiveSDCam" -addext "subjectAltName=IP:<your-lan-ip>,IP:127.0.0.1,DNS:localhost"
```

Then open `https://<your-lan-ip>:7860/live` on the phone (accept the cert warning once).

## Performance (5070 Ti)

The live engines are optimized with **TAESD** (fast tiny VAE), **channels_last**, flash-SDP,
and an **unsharp** pass to keep faces crisp. **Lite** additionally uses `torch.compile`
(via `triton-windows`) → ~**16 fps** at 512². The inpaint modes (Wardrobe/Try-On) run ~4 fps
live; `torch.compile` is disabled there (it FX-trace-errors on the inpaint pipeline). HQ
**Capture** swaps back to the full VAE for maximum sharpness.

A **single-flight lock** drops frames while the GPU is busy (no lag buildup), and the `/live`
loop self-paces to the GPU.

| File | Role |
|---|---|
| `app.py` | Gradio UI + `/live`, `/infer`, `/capture`, `/caption`, `/probe` routes; FastAPI/uvicorn launch |
| `engine.py` | Lite engine (SD-Turbo img2img) |
| `engine_wardrobe.py` | Wardrobe engine (clothes parser + SD1.5 LCM-inpaint, live + HQ capture) |
| `engine_tryon.py` | Try-On live engine (IP-Adapter garment conditioning) |
| `engine_catvton.py` | Try-On HQ capture (CatVTON, on-demand GPU; uses `third_party/CatVTON`) |
| `captioner.py` | BLIP garment captioner (auto-describes uploaded garments) |
| `segmenter.py` | Person segmenter + background composite (Keep background) |
| `perf.py` | Shared GPU optimizations (TAESD, channels_last, torch.compile, sharpen) |
| `download_models.py` | Pre-fetch + validate all required models |
| `check_gpu.py` | GPU / CUDA sanity check |
| `*smoke*`, `*test*`, `*bench*`, `spike_*` | Standalone validation/benchmark scripts |

## Models used

- [`stabilityai/sd-turbo`](https://huggingface.co/stabilityai/sd-turbo) — Lite img2img
- [`Lykon/dreamshaper-8-inpainting`](https://huggingface.co/Lykon/dreamshaper-8-inpainting) + [`latent-consistency/lcm-lora-sdv1-5`](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) — Wardrobe / Try-On inpaint
- [`h94/IP-Adapter`](https://huggingface.co/h94/IP-Adapter) — Try-On live garment conditioning
- [`zhengchong/CatVTON`](https://huggingface.co/zhengchong/CatVTON) — Try-On HQ capture (faithful try-on)
- [`mattmdjaga/segformer_b2_clothes`](https://huggingface.co/mattmdjaga/segformer_b2_clothes) — garment mask
- [`Salesforce/blip-image-captioning-large`](https://huggingface.co/Salesforce/blip-image-captioning-large) — garment captioning
- [`madebyollin/taesd`](https://huggingface.co/madebyollin/taesd) — fast VAE decode
- torchvision DeepLabV3-MobileNet — person segmentation

## Notes

- Live transforms flicker slightly frame-to-frame (per-frame diffusion); **Stabilize** (fixed
  seed) + **Temporal smoothing** reduce it.
- Live previews are approximate; the faithful result is the **📸 Capture**.
- CatVTON (`third_party/CatVTON`) and the TLS cert/key are git-ignored.
