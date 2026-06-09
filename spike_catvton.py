"""Spike: CatVTON faithful try-on on our torch 2.11 / diffusers 0.37 stack.
Feeds our own segformer garment mask (no DensePose). Person + garment -> try-on.
"""
import sys, time
sys.path.insert(0, "third_party/CatVTON")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

from model.pipeline import CatVTONPipeline

DEVICE, DTYPE = "cuda", torch.bfloat16
GARMENT_CLASSES = (4, 5, 6, 7, 8, 17)

# our mask (dilated a bit; CatVTON likes a cloth-agnostic region)
seg_proc = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
seg_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to(DEVICE).eval()

@torch.no_grad()
def garment_mask(pil, size, dilate=15):
    inp = seg_proc(images=pil, return_tensors="pt").to(DEVICE)
    up = F.interpolate(seg_model(**inp).logits, size=size[::-1], mode="bilinear", align_corners=False)
    m = np.isin(up.argmax(1)[0].cpu().numpy(), GARMENT_CLASSES).astype(np.uint8) * 255
    m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
    return Image.fromarray(m)

print("loading CatVTON pipeline...")
t0 = time.time()
pipe = CatVTONPipeline(
    base_ckpt="booksforcharlie/stable-diffusion-inpainting",
    attn_ckpt="zhengchong/CatVTON",
    attn_ckpt_version="mix",
    weight_dtype=DTYPE,
    device=DEVICE,
    skip_safety_check=True,
)
print(f"loaded in {time.time()-t0:.1f}s")

H, W = 1024, 768
person = Image.open("test_before.png").convert("RGB")
garment = Image.open("garment_suit.png").convert("RGB")
# mask must match the person's size (CatVTON asserts before its own internal resize)
mask = garment_mask(person, person.size)

t = time.time()
result = pipe(image=person, condition_image=garment, mask=mask,
              num_inference_steps=30, guidance_scale=2.5, height=H, width=W,
              generator=torch.Generator(DEVICE).manual_seed(42))
torch.cuda.synchronize()
gen_s = time.time() - t
result = result[0] if isinstance(result, list) else result
result.save("catvton_result.png")

combo = Image.new("RGB", (W*3+20, H), "white")
for i, im in enumerate([person.resize((W, H)), garment.resize((W, H)), result.resize((W, H))]):
    combo.paste(im, (i*(W+10), 0))
combo.save("catvton_combo.png")
print(f"\nCatVTON: {gen_s:.1f}s/image (30 steps, {W}x{H})")
print("saved catvton_combo.png  [person | garment | try-on result]")
