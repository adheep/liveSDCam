"""Spike: IP-Adapter garment conditioning on SD1.5 LCM-inpaint.
Feeds a garment image as an appearance reference into the clothing-region inpaint.
Validates it runs (IP-Adapter + LCM together) and measures speed.
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
GARMENT_CLASSES = (4, 5, 6, 7, 8, 17)

# --- mask ---
seg_proc = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
seg_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to(DEVICE).eval()

@torch.no_grad()
def garment_mask(pil, dilate=7, feather=9):
    inp = seg_proc(images=pil, return_tensors="pt").to(DEVICE)
    up = F.interpolate(seg_model(**inp).logits, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
    m = np.isin(up.argmax(1)[0].cpu().numpy(), GARMENT_CLASSES).astype(np.uint8) * 255
    m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
    return Image.fromarray(cv2.GaussianBlur(m, (feather, feather), 0))

# --- pipeline + IP-Adapter ---
print("loading inpaint + LCM + IP-Adapter...")
pipe = AutoPipelineForInpainting.from_pretrained(
    "Lykon/dreamshaper-8-inpainting", torch_dtype=DTYPE, safety_checker=None).to(DEVICE)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe.fuse_lora()
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.safetensors")
pipe.set_ip_adapter_scale(0.7)
pipe.set_progress_bar_config(disable=True)

person = Image.open("test_before.png").convert("RGB").resize((SIZE, SIZE))
garment = Image.open("garment_suit.png").convert("RGB").resize((SIZE, SIZE))
mask = garment_mask(person)

def run(steps=4):
    return pipe(prompt="a person wearing this outfit, detailed clothing, photorealistic",
                negative_prompt="blurry, deformed, low quality",
                image=person, mask_image=mask, ip_adapter_image=garment,
                num_inference_steps=steps, guidance_scale=1.5, strength=1.0,
                generator=torch.Generator(DEVICE).manual_seed(42)).images[0]

run()  # warmup
ts = []
for _ in range(6):
    t = time.time(); run(); torch.cuda.synchronize(); ts.append(time.time()-t)
ts.sort(); ms = ts[len(ts)//2]*1000
out = run()

combo = Image.new("RGB", (SIZE*4+30, SIZE), "white")
for i, im in enumerate([person, garment, mask.convert("RGB"), out]):
    combo.paste(im, (i*(SIZE+10), 0))
combo.save("spike_ipadapter_combo.png")
print(f"\nIP-Adapter inpaint: {ms:.0f} ms -> {1000/ms:.1f} fps")
print("saved spike_ipadapter_combo.png  [person | garment | mask | result]")
