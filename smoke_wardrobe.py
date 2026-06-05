"""Phase 'Wardrobe' smoke test: clothing parse + SD1.5 LCM-inpaint.
Masks ONLY the garment (face/hands/hair preserved) and repaints it from a prompt.
Validates quality + measures live(4-step) vs capture(8-step) speed.
"""
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from diffusers import AutoPipelineForInpainting, LCMScheduler
from diffusers.utils import logging as dl

dl.set_verbosity_error()
DEVICE, DTYPE, SIZE = "cuda", torch.float16, 512

# ---- clothing parser (ATR labels) ----
# 4=Upper-clothes 5=Skirt 6=Pants 7=Dress 8=Belt 17=Scarf ; 2=Hair 11=Face 14/15=Arms
GARMENT_CLASSES = {4, 5, 6, 7, 8, 17}
print("Loading clothes parser (segformer_b2_clothes)...")
seg_proc = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
seg_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to(DEVICE).eval()


@torch.no_grad()
def garment_mask(pil_img, dilate=7, feather=9):
    inputs = seg_proc(images=pil_img, return_tensors="pt").to(DEVICE)
    logits = seg_model(**inputs).logits                       # (1,C,h,w)
    up = F.interpolate(logits, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
    labels = up.argmax(1)[0].cpu().numpy()
    m = np.isin(labels, list(GARMENT_CLASSES)).astype(np.uint8) * 255
    if dilate:
        m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8), iterations=1)
    if feather:
        m = cv2.GaussianBlur(m, (feather, feather), 0)
    return Image.fromarray(m)


# ---- SD1.5 LCM inpaint ----
print("Loading SD1.5 inpaint (dreamshaper-8-inpainting) + LCM-LoRA...")
pipe = AutoPipelineForInpainting.from_pretrained(
    "Lykon/dreamshaper-8-inpainting", torch_dtype=DTYPE, safety_checker=None,
).to(DEVICE)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.fuse_lora()
pipe.set_progress_bar_config(disable=True)

src = Image.open("test_before.png").convert("RGB").resize((SIZE, SIZE))
mask = garment_mask(src)
mask.save("wardrobe_mask.png")

NEG = "blurry, distorted, deformed, extra limbs, low quality, naked"
def inpaint(prompt, steps):
    return pipe(prompt=prompt, negative_prompt=NEG, image=src, mask_image=mask,
                num_inference_steps=steps, guidance_scale=1.5, strength=1.0,
                generator=torch.Generator(DEVICE).manual_seed(42)).images[0]

PROMPTS = {
    "suit": "wearing an elegant tailored business suit and tie, formal, detailed fabric",
    "kurta": "wearing a traditional white kurta, indian ethnic wear, detailed fabric",
}

# warmup + timing (parse + 4-step inpaint)
inpaint(PROMPTS["suit"], 4)
t = time.time(); garment_mask(src); torch.cuda.synchronize(); parse_ms = (time.time()-t)*1000
ts = []
for _ in range(8):
    t = time.time(); inpaint(PROMPTS["suit"], 4); torch.cuda.synchronize(); ts.append(time.time()-t)
ts.sort(); live_ms = ts[len(ts)//2]*1000
t = time.time(); inpaint(PROMPTS["suit"], 8); torch.cuda.synchronize(); cap_ms = (time.time()-t)*1000

outs = {k: inpaint(p, 8) for k, p in PROMPTS.items()}
panels = [("SOURCE", src), ("MASK", mask.convert("RGB")),
          ("suit", outs["suit"]), ("kurta", outs["kurta"])]
combo = Image.new("RGB", (SIZE*len(panels)+10*(len(panels)-1), SIZE), "white")
for i, (_, im) in enumerate(panels):
    combo.paste(im, (i*(SIZE+10), 0))
combo.save("wardrobe_combo.png")

vram = torch.cuda.max_memory_allocated()/1024**3
print(f"\n=== WARDROBE (SD1.5 LCM-inpaint) ===")
print(f"clothes parse      : {parse_ms:6.0f} ms")
print(f"live  (4 steps)    : {live_ms:6.0f} ms  total ~{parse_ms+live_ms:.0f}ms -> {1000/(parse_ms+live_ms):.1f} fps")
print(f"capture (8 steps)  : {cap_ms:6.0f} ms")
print(f"VRAM {vram:.2f} GB | saved wardrobe_mask.png, wardrobe_combo.png")
