"""Phase 0 smoke test: SD-Turbo img2img latency on this GPU.
Generates a synthetic input frame, runs img2img repeatedly, reports fps.
"""
import time
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import logging as dl

dl.set_verbosity_error()

DEVICE = "cuda"
DTYPE = torch.float16
SIZE = 512
STEPS = 2          # SD-Turbo works in 1-4 steps
STRENGTH = 0.5     # how far from the source frame
PROMPT = "a disney pixar style character portrait, vibrant, detailed"

print("Loading SD-Turbo (first run downloads ~2.5GB)...")
t0 = time.time()
pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=DTYPE,
    variant="fp16",
    safety_checker=None,
)
pipe = pipe.to(DEVICE)
pipe.set_progress_bar_config(disable=True)
print(f"Model loaded in {time.time()-t0:.1f}s")

# Synthetic input image (stand-in for a webcam frame)
init = torch.rand(1, 3, SIZE, SIZE, device=DEVICE, dtype=DTYPE)
from torchvision.transforms.functional import to_pil_image
init_img = to_pil_image(init[0].float().cpu())

gen = torch.Generator(device=DEVICE).manual_seed(42)

def run_one():
    return pipe(
        prompt=PROMPT,
        image=init_img,
        num_inference_steps=STEPS,
        strength=STRENGTH,
        guidance_scale=0.0,   # turbo models use CFG 0
        generator=gen,
    ).images[0]

print("\nWarmup (3 iters, includes CUDA kernel compile)...")
for _ in range(3):
    run_one()
torch.cuda.synchronize()

print("Timing 15 iters...")
times = []
for _ in range(15):
    t = time.time()
    run_one()
    torch.cuda.synchronize()
    times.append(time.time() - t)

times.sort()
median = times[len(times)//2]
avg = sum(times) / len(times)
vram = torch.cuda.max_memory_allocated() / 1024**3

print(f"\n=== RESULTS (size={SIZE}, steps={STEPS}, strength={STRENGTH}) ===")
print(f"median : {median*1000:6.1f} ms  ->  {1/median:5.1f} fps")
print(f"avg    : {avg*1000:6.1f} ms  ->  {1/avg:5.1f} fps")
print(f"best   : {times[0]*1000:6.1f} ms  ->  {1/times[0]:5.1f} fps")
print(f"peak VRAM: {vram:.2f} GB")
print("\n=== GO: img2img runs on GPU ===")
