"""Phase 3 smoke test: SD-Turbo + SD2.1 Canny ControlNet img2img.
Verifies the pipeline loads/runs and measures the latency hit vs plain img2img.
"""
import time
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from diffusers.utils import logging as dl

dl.set_verbosity_error()

DEVICE, DTYPE, SIZE = "cuda", torch.float16, 512
STEPS, STRENGTH = 2, 0.6
CTRL_SCALE = 0.8
PROMPT = "a disney pixar 3d animated character, big expressive eyes, vibrant colors"
NEG = "blurry, distorted, deformed, ugly"
CN_ID = "thibaud/controlnet-sd21-canny-diffusers"

print("Loading Canny ControlNet (SD2.1)...")
controlnet = ControlNetModel.from_pretrained(CN_ID, torch_dtype=DTYPE)
print("Loading SD-Turbo + ControlNet pipeline...")
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "stabilityai/sd-turbo", controlnet=controlnet, torch_dtype=DTYPE,
    variant="fp16", safety_checker=None,
).to(DEVICE)
pipe.set_progress_bar_config(disable=True)


def prep_square(img, size):
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return img.resize((size, size), Image.BILINEAR)


def canny(pil_img):
    arr = np.array(pil_img)
    edges = cv2.Canny(arr, 100, 200)
    edges = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges)


# use the real webcam frame captured earlier (avoids camera conflict with running app)
import os
if os.path.exists("test_before.png"):
    src = prep_square(Image.open("test_before.png").convert("RGB"), SIZE)
else:
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    for _ in range(8):
        cap.read(); time.sleep(0.03)
    ok, frame = cap.read()
    cap.release()
    assert ok, "webcam read failed (close the browser tab using the camera)"
    src = prep_square(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), SIZE)
ctrl = canny(src)
src.save("cn_before.png")
ctrl.save("cn_canny.png")

gen = lambda: torch.Generator(DEVICE).manual_seed(1234)


def run():
    return pipe(
        prompt=PROMPT, negative_prompt=NEG, image=src, control_image=ctrl,
        num_inference_steps=STEPS, strength=STRENGTH, guidance_scale=0.0,
        controlnet_conditioning_scale=CTRL_SCALE, generator=gen(),
    ).images[0]


print("Warmup...")
for _ in range(3):
    run()
torch.cuda.synchronize()

print("Timing 12 iters...")
ts = []
for _ in range(12):
    t = time.time(); run(); torch.cuda.synchronize(); ts.append(time.time() - t)
ts.sort()
med = ts[len(ts) // 2]
out = run()
out.save("cn_after.png")

combo = Image.new("RGB", (SIZE * 3 + 20, SIZE), "white")
combo.paste(src, (0, 0)); combo.paste(ctrl, (SIZE + 10, 0)); combo.paste(out, (SIZE * 2 + 20, 0))
combo.save("cn_combo.png")

vram = torch.cuda.max_memory_allocated() / 1024**3
print(f"\n=== ControlNet img2img: {med*1000:.0f} ms -> {1/med:.1f} fps | VRAM {vram:.2f} GB ===")
print("Saved cn_before.png / cn_canny.png / cn_after.png / cn_combo.png")
