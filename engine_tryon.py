"""LiveSD Try-On (live) engine: IP-Adapter garment-conditioned inpaint.

You attach a garment image; it conditions the clothing-region inpaint so the
live feed shows you in something matching that garment's color/material/style
(not an exact reproduction -- that's the CatVTON Capture path). Face / pose /
background preserved. ~2-3 fps. Lazy-loaded on first use.
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
GARMENT_CLASSES = (4, 5, 6, 7, 8, 17)
NEG_PROMPT = "blurry, distorted, deformed, extra limbs, low quality, naked, nsfw"
LIVE_STEPS = 4


class TryOnEngine:
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
        print("[tryon] loading clothes parser ...")
        t0 = time.time()
        self.seg_proc = SegformerImageProcessor.from_pretrained(PARSER_ID)
        self.seg_model = AutoModelForSemanticSegmentation.from_pretrained(PARSER_ID).to(DEVICE).eval()
        print("[tryon] loading SD1.5 inpaint + LCM-LoRA + IP-Adapter ...")
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            INPAINT_ID, torch_dtype=DTYPE, safety_checker=None,
        ).to(DEVICE)
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.load_lora_weights(LCM_LORA_ID)
        self.pipe.fuse_lora()
        self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                                  weight_name="ip-adapter_sd15.safetensors")
        self.pipe.set_progress_bar_config(disable=True)
        from perf import optimize_pipe
        # torch.compile breaks the inpaint pipeline (FX-trace error) -> compile off; TAESD still helps.
        optimize_pipe(self.pipe, DTYPE, DEVICE, use_taesd=True, compile_unet=False, label=":tryon")
        print(f"[tryon] loaded in {time.time()-t0:.1f}s; warming up (compiling)...")
        self._warmup()
        self.loaded = True
        print("[tryon] ready.")

    def _warmup(self):
        dummy = Image.new("RGB", (SIZE, SIZE), (120, 120, 120))
        mask = Image.new("L", (SIZE, SIZE), 255)
        self.pipe.set_ip_adapter_scale(0.7)
        for _ in range(4):  # extra iters so torch.compile finishes before serving
            self._infer(dummy, mask, dummy, "a person", LIVE_STEPS, 0.7, True)
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

    def _infer(self, pil_img, mask_img, garment_img, prompt, steps, ip_scale, fixed_seed, negative=None):
        self.pipe.set_ip_adapter_scale(float(ip_scale))
        generator = torch.Generator(DEVICE).manual_seed(1234) if fixed_seed else None
        out = self.pipe(
            prompt=prompt or "a person wearing this outfit, detailed clothing, photorealistic",
            negative_prompt=negative or NEG_PROMPT,
            image=pil_img,
            mask_image=mask_img,
            ip_adapter_image=garment_img,
            num_inference_steps=int(steps),
            guidance_scale=1.5,
            strength=1.0,
            generator=generator,
        ).images[0]
        return np.array(out)

    def reset_temporal(self):
        self._smooth_prev = None

    def transform(self, frame, garment, prompt, steps, stabilize, smoothing, ip_scale, negative=None):
        """Live preview. `garment` is an RGB np array (the attached garment image).
        Returns (output_rgb_np, fps_text). Drops frame if busy or no garment yet."""
        if not valid_frame(frame) or not valid_frame(garment):
            return self._last_out, self._fps_text()
        if not self._lock.acquire(blocking=False):
            return self._last_out, self._fps_text()
        try:
            pil = self._prep(frame)
            garment_pil = Image.fromarray(garment).convert("RGB").resize((SIZE, SIZE))
            mask = self._garment_mask(pil)
            steps = int(steps) if steps else LIVE_STEPS
            out = self._infer(pil, mask, garment_pil, prompt, steps, ip_scale, stabilize, negative)

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
            return "warming up..." if self.loaded else "attach a garment image to begin"
        return f"{self._fps_ema:4.1f} fps   ({self._frames_done} frames)  [Try-On live]"
