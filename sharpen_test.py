"""Does a cheap unsharp-mask recover TAESD sharpness? Compare against full VAE."""
import numpy as np
import torch, cv2
from PIL import Image, ImageDraw
from diffusers import AutoPipelineForImage2Image, AutoencoderTiny

DEVICE, DTYPE, SIZE = "cuda", torch.float16, 512
pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sd-turbo", torch_dtype=DTYPE, variant="fp16", safety_checker=None).to(DEVICE)
pipe.set_progress_bar_config(disable=True)
full_vae = pipe.vae
taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=DTYPE).to(DEVICE)
src = Image.open("test_before.png").convert("RGB").resize((SIZE, SIZE))
PROMPT = "a disney pixar 3d animated character, big expressive eyes, vibrant colors, cinematic lighting"

def run():
    return np.array(pipe(prompt=PROMPT, negative_prompt="blurry, low quality", image=src,
        num_inference_steps=2, strength=0.6, guidance_scale=0.0,
        generator=torch.Generator(DEVICE).manual_seed(42)).images[0])

def unsharp(arr, amount, sigma=1.2):
    blur = cv2.GaussianBlur(arr, (0, 0), sigma)
    return np.clip(arr.astype(np.float32)*(1+amount) - blur.astype(np.float32)*amount, 0, 255).astype(np.uint8)

pipe.vae = taesd
t = run()
pipe.vae = full_vae
f = run()

panels = [("TAESD raw", t), ("TAESD +sharp 0.6", unsharp(t,0.6)),
          ("TAESD +sharp 1.0", unsharp(t,1.0)), ("Full VAE", f)]
grid = Image.new("RGB", (SIZE*4+30, SIZE+24), "white"); d = ImageDraw.Draw(grid)
for i,(lab,im) in enumerate(panels):
    x=i*(SIZE+10); grid.paste(Image.fromarray(im),(x,24)); d.text((x+4,6),lab,fill="black")
grid.save("sharpen_compare.png"); print("saved sharpen_compare.png")
