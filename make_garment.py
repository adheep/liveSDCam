"""Generate a couple of flat-lay garment images to test the try-on spikes."""
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.utils import logging as dl

dl.set_verbosity_error()
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16", safety_checker=None
).to("cuda")
pipe.set_progress_bar_config(disable=True)

GARMENTS = {
    "garment_suit.png": "a flat lay product photo of a navy blue pinstripe business suit jacket with a red tie, centered, plain white background, e-commerce",
    "garment_kurta.png": "a flat lay product photo of a white embroidered traditional indian kurta, centered, plain white background, e-commerce",
}
for fname, prompt in GARMENTS.items():
    img = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0,
               generator=torch.Generator("cuda").manual_seed(7)).images[0]
    img.save(fname)
    print("saved", fname)
