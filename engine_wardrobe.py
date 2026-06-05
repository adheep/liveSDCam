"""LiveSD Wardrobe engine: change ONLY the clothes, keep the character.

Parses the garment region (face/hair/hands excluded) and inpaints just that
region from a text prompt using SD1.5 + LCM-LoRA (few-step, fast). Identity,
pose and background are the original pixels -- untouched by construction.

Two paths:
  transform() - live preview, few steps (~3 fps), temporal smoothing
  capture()   - one-shot high-step render for a clean, shareable result

Lazy-loaded on first use.
"""
import threading
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from diffusers import AutoPipelineForInpainting, LCMScheduler
from diffusers.utils import logging as dl

from segmenter import valid_frame

dl.set_verbosity_error()

DEVICE = "cuda"
DTYPE = torch.float16
INPAINT_ID = "Lykon/dreamshaper-8-inpainting"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"
PARSER_ID = "mattmdjaga/segformer_b2_clothes"
SIZE = 512
# ATR labels: 4=Upper-clothes 5=Skirt 6=Pants 7=Dress 8=Belt 17=Scarf
GARMENT_CLASSES = (4, 5, 6, 7, 8, 17)
NEG_PROMPT = "blurry, distorted, deformed, extra limbs, low quality, naked, nsfw"
LIVE_STEPS = 4
CAPTURE_STEPS = 10


class WardrobeEngine:
    def __init__(self):
        self.pipe = None
        self.seg_proc = None
        self.seg_model = None
        self.loaded = False
        self._lock = threading.Lock()
        self._last_out = None
        self._smooth_prev = None
        self._fps_ema = None
        self._last_ts = None
        self._frames_done = 0

    def load(self):
        if self.loaded:
            return
        print("[wardrobe] loading clothes parser ...")
        t0 = time.time()
        self.seg_proc = SegformerImageProcessor.from_pretrained(PARSER_ID)
        self.seg_model = AutoModelForSemanticSegmentation.from_pretrained(PARSER_ID).to(DEVICE).eval()
        print("[wardrobe] loading SD1.5 inpaint + LCM-LoRA ...")
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            INPAINT_ID, torch_dtype=DTYPE, safety_checker=None,
        ).to(DEVICE)
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.load_lora_weights(LCM_LORA_ID)
        self.pipe.fuse_lora()
        self.pipe.set_progress_bar_config(disable=True)
        print(f"[wardrobe] loaded in {time.time()-t0:.1f}s; warming up...")
        self._warmup()
        self.loaded = True
        print("[wardrobe] ready.")

    def _warmup(self):
        dummy = Image.new("RGB", (SIZE, SIZE), (120, 120, 120))
        mask = Image.new("L", (SIZE, SIZE), 255)
        for _ in range(2):
            self._inpaint(dummy, mask, "a shirt", NEG_PROMPT, LIVE_STEPS, True)
        torch.cuda.synchronize()

    @staticmethod
    def _prep(frame: np.ndarray) -> Image.Image:
        img = Image.fromarray(frame)
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        return img.resize((SIZE, SIZE), Image.BILINEAR)

    @torch.no_grad()
    def _garment_mask(self, pil_img, dilate=7, feather=9) -> Image.Image:
        inputs = self.seg_proc(images=pil_img, return_tensors="pt").to(DEVICE)
        logits = self.seg_model(**inputs).logits
        up = F.interpolate(logits, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        labels = up.argmax(1)[0].cpu().numpy()
        m = np.isin(labels, GARMENT_CLASSES).astype(np.uint8) * 255
        if dilate:
            m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8), iterations=1)
        if feather:
            m = cv2.GaussianBlur(m, (feather, feather), 0)
        return Image.fromarray(m)

    def _inpaint(self, pil_img, mask_img, prompt, negative, steps, fixed_seed):
        generator = torch.Generator(DEVICE).manual_seed(1234) if fixed_seed else None
        out = self.pipe(
            prompt=prompt or "clothing",
            negative_prompt=negative or NEG_PROMPT,
            image=pil_img,
            mask_image=mask_img,
            num_inference_steps=int(steps),
            guidance_scale=1.5,
            strength=1.0,
            generator=generator,
        ).images[0]
        return np.array(out)

    def reset_temporal(self):
        self._smooth_prev = None

    def transform(self, frame, prompt, steps, stabilize, smoothing, negative=None):
        """Live preview. Returns (output_rgb_np, fps_text). Drops frame if busy."""
        if not valid_frame(frame):
            return self._last_out, self._fps_text()
        if not self._lock.acquire(blocking=False):
            return self._last_out, self._fps_text()
        try:
            pil = self._prep(frame)
            mask = self._garment_mask(pil)
            steps = int(steps) if steps else LIVE_STEPS
            out = self._inpaint(pil, mask, prompt, negative, steps, stabilize)

            a = float(smoothing)
            if a > 0.0 and self._smooth_prev is not None:
                out = (a * self._smooth_prev.astype(np.float32)
                       + (1.0 - a) * out.astype(np.float32)).astype(np.uint8)
            self._smooth_prev = out

            self._tick_fps()
            self._last_out = out
            return out, self._fps_text()
        finally:
            self._lock.release()

    def capture(self, frame, prompt, negative=None, steps=CAPTURE_STEPS):
        """One-shot high-quality render of the current frame (blocks until done)."""
        if not valid_frame(frame):
            return self._last_out
        with self._lock:
            pil = self._prep(frame)
            mask = self._garment_mask(pil, dilate=9, feather=11)
            out = self._inpaint(pil, mask, prompt, negative, steps, fixed_seed=True)
            return out

    def _tick_fps(self):
        now = time.time()
        if self._last_ts is not None:
            dt = now - self._last_ts
            if dt > 0:
                inst = 1.0 / dt
                self._fps_ema = inst if self._fps_ema is None else 0.85 * self._fps_ema + 0.15 * inst
        self._last_ts = now
        self._frames_done += 1

    def _fps_text(self):
        if self._fps_ema is None:
            return "warming up..." if self.loaded else "not loaded"
        return f"{self._fps_ema:4.1f} fps   ({self._frames_done} frames)  [Wardrobe]"
