"""BLIP garment captioner.

When a garment image is uploaded in Try-On mode, this describes it (e.g.
"a person wearing a blue embroidered kurta") so the text prompt AGREES with the
IP-Adapter garment image instead of fighting it. Runs once per upload, lazy-loaded.
"""
import threading

import numpy as np
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

DEVICE = "cuda"
DTYPE = torch.float16
MODEL_ID = "Salesforce/blip-image-captioning-large"
# conditional prefix steers BLIP to describe the worn garment, not the background
PREFIX = "a person wearing"
SUFFIX = ", photorealistic, detailed fabric, natural lighting"


class Captioner:
    def __init__(self):
        self.proc = None
        self.model = None
        self.loaded = False
        self._lock = threading.Lock()

    def load(self):
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            print(f"[caption] loading BLIP ({MODEL_ID}) ...")
            self.proc = BlipProcessor.from_pretrained(MODEL_ID)
            self.model = BlipForConditionalGeneration.from_pretrained(
                MODEL_ID, torch_dtype=DTYPE).to(DEVICE).eval()
            self.loaded = True
            print("[caption] ready.")

    @torch.no_grad()
    def caption_garment(self, frame) -> str:
        """frame: RGB np array of the garment. Returns a prompt describing it."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return ""
        self.load()
        pil = Image.fromarray(frame).convert("RGB")
        inputs = self.proc(pil, text=PREFIX, return_tensors="pt").to(DEVICE)
        inputs["pixel_values"] = inputs["pixel_values"].to(DTYPE)
        out = self.model.generate(**inputs, max_new_tokens=30, num_beams=3)
        caption = self.proc.decode(out[0], skip_special_tokens=True).strip()
        if not caption:
            return ""
        return caption + SUFFIX


captioner = Captioner()
