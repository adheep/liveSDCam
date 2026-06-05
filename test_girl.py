"""Test the 'Girl' prompt at increasing strength/steps to find what actually feminizes."""
import torch
import numpy as np
from PIL import Image, ImageDraw
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import logging as dl

dl.set_verbosity_error()
DEVICE, DTYPE, SIZE = "cuda", torch.float16, 512
PROMPT = "a beautiful young woman, feminine face, long flowing hair, soft delicate features, natural makeup, portrait, detailed, photorealistic"
NEG = "man, male, beard, mustache, masculine, blurry, distorted, deformed"

pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sd-turbo", torch_dtype=DTYPE, variant="fp16", safety_checker=None
).to(DEVICE)
pipe.set_progress_bar_config(disable=True)

src = Image.open("test_before.png").convert("RGB").resize((SIZE, SIZE))
configs = [(0.6, 2), (0.75, 2), (0.85, 3), (0.95, 4)]
imgs = [src]
for strength, steps in configs:
    out = pipe(prompt=PROMPT, negative_prompt=NEG, image=src,
               num_inference_steps=steps, strength=strength, guidance_scale=0.0,
               generator=torch.Generator(DEVICE).manual_seed(42)).images[0]
    imgs.append(out)

labels = ["SOURCE"] + [f"s={s} steps={st}" for s, st in configs]
combo = Image.new("RGB", (SIZE * len(imgs) + 10 * (len(imgs) - 1), SIZE + 24), "white")
d = ImageDraw.Draw(combo)
for i, (im, lab) in enumerate(zip(imgs, labels)):
    x = i * (SIZE + 10)
    combo.paste(im, (x, 24))
    d.text((x + 4, 4), lab, fill="black")
combo.save("girl_strengths.png")
print("saved girl_strengths.png")
