"""Visual test: capture ONE webcam frame, run img2img, save before/after.
Falls back to a txt2img-generated portrait if no webcam is available.
"""
import time
import torch
import numpy as np
from PIL import Image
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import logging as dl

dl.set_verbosity_error()

DEVICE, DTYPE, SIZE = "cuda", torch.float16, 512
STEPS, STRENGTH = 2, 0.6
PROMPT = "a disney pixar 3d animated character, big expressive eyes, vibrant colors, cinematic lighting"
NEG = "blurry, distorted, ugly, deformed"


def center_crop_square(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return img.resize((size, size), Image.LANCZOS)


def grab_webcam_frame():
    import cv2
    # CAP_DSHOW = DirectShow backend, most reliable on Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None
    # discard a few frames so auto-exposure settles
    for _ in range(8):
        cap.read()
        time.sleep(0.05)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


print("Loading SD-Turbo...")
i2i = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sd-turbo", torch_dtype=DTYPE, variant="fp16", safety_checker=None
).to(DEVICE)
i2i.set_progress_bar_config(disable=True)

# --- get a source image ---
print("Trying webcam...")
src = grab_webcam_frame()
if src is not None:
    print(f"  Captured webcam frame: {src.size}")
    source = "webcam"
else:
    print("  No webcam -> generating a synthetic portrait via txt2img instead")
    from diffusers import AutoPipelineForText2Image
    t2i = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo", torch_dtype=DTYPE, variant="fp16", safety_checker=None
    ).to(DEVICE)
    t2i.set_progress_bar_config(disable=True)
    src = t2i(prompt="a realistic photo portrait of a person, front facing, neutral background",
              num_inference_steps=2, guidance_scale=0.0,
              generator=torch.Generator(DEVICE).manual_seed(7)).images[0]
    source = "synthetic"

src = center_crop_square(src, SIZE)
src.save("test_before.png")

# --- transform ---
print(f"Running img2img  (prompt='{PROMPT[:40]}...', strength={STRENGTH})")
t0 = time.time()
out = i2i(prompt=PROMPT, negative_prompt=NEG, image=src,
          num_inference_steps=STEPS, strength=STRENGTH, guidance_scale=0.0,
          generator=torch.Generator(DEVICE).manual_seed(42)).images[0]
torch.cuda.synchronize()
print(f"  done in {(time.time()-t0)*1000:.0f} ms")
out.save("test_after.png")

# --- side-by-side combined image ---
combo = Image.new("RGB", (SIZE * 2 + 10, SIZE), "white")
combo.paste(src, (0, 0))
combo.paste(out, (SIZE + 10, 0))
combo.save("test_combo.png")

print(f"\nSaved (source={source}):")
print("  test_before.png  (input)")
print("  test_after.png   (transformed)")
print("  test_combo.png   (side by side)")
