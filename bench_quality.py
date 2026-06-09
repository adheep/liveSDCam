"""Compare TAESD vs full VAE (and 2 vs 3 steps) for Lite: sharpness + fps."""
import time
import numpy as np
import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image, AutoencoderTiny

DEVICE, DTYPE, SIZE = "cuda", torch.float16, 512
torch.backends.cuda.matmul.allow_tf32 = True
try: torch.backends.cuda.enable_flash_sdp(True)
except Exception: pass

pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sd-turbo", torch_dtype=DTYPE, variant="fp16", safety_checker=None).to(DEVICE)
pipe.set_progress_bar_config(disable=True)
pipe.unet.to(memory_format=torch.channels_last)
full_vae = pipe.vae
full_vae.to(memory_format=torch.channels_last)
taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=DTYPE).to(DEVICE)
pipe.unet = torch.compile(pipe.unet, fullgraph=False)

src = Image.open("test_before.png").convert("RGB").resize((SIZE, SIZE))
PROMPT = "a disney pixar 3d animated character, big expressive eyes, vibrant colors, cinematic lighting"

def run(steps):
    return pipe(prompt=PROMPT, negative_prompt="blurry, low quality", image=src,
               num_inference_steps=steps, strength=0.6, guidance_scale=0.0,
               generator=torch.Generator(DEVICE).manual_seed(42)).images[0]

def bench(label, vae, steps):
    pipe.vae = vae
    for _ in range(4): run(steps)         # warmup / compile
    torch.cuda.synchronize()
    ts = []
    for _ in range(12):
        t = time.time(); out = run(steps); torch.cuda.synchronize(); ts.append(time.time()-t)
    ts.sort(); med = ts[len(ts)//2]
    print(f"{label:28} {med*1000:6.1f} ms  ->  {1/med:4.1f} fps")
    return out

panels = []
panels.append(("source", src))
panels.append(("TAESD 2-step (current)", bench("TAESD 2-step", taesd, 2)))
panels.append(("Full VAE 2-step", bench("Full VAE 2-step", full_vae, 2)))
panels.append(("Full VAE 3-step", bench("Full VAE 3-step", full_vae, 3)))

from PIL import ImageDraw
n = len(panels)
grid = Image.new("RGB", (SIZE*n + 10*(n-1), SIZE+24), "white")
d = ImageDraw.Draw(grid)
for i,(lab,im) in enumerate(panels):
    x = i*(SIZE+10); grid.paste(im.resize((SIZE,SIZE)), (x,24)); d.text((x+4,6), lab, fill="black")
grid.save("quality_compare.png")
print("saved quality_compare.png")
