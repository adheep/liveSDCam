"""LiveSD Advanced engine (Phase 3): SD-Turbo + Canny ControlNet img2img
with temporal output smoothing for frame-to-frame consistency.

Structure-locks each frame to the live webcam edges (keeps you recognizable
and stable), then blends consecutive outputs to remove residual flicker.
Lazy-loaded only when the user switches to Advanced mode.
"""
import threading
import time

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from diffusers.utils import logging as dl

dl.set_verbosity_error()

DEVICE = "cuda"
DTYPE = torch.float16
MODEL_ID = "stabilityai/sd-turbo"
CONTROLNET_ID = "thibaud/controlnet-sd21-canny-diffusers"
SIZE = 512
NEG_PROMPT = "blurry, distorted, deformed, ugly, low quality, extra limbs"


class AdvancedEngine:
    def __init__(self):
        self.pipe = None
        self.loaded = False
        self._lock = threading.Lock()
        self._last_out = None          # last successful output (returned when busy)
        self._smooth_prev = None       # previous output for temporal EMA
        self._fps_ema = None
        self._last_ts = None
        self._frames_done = 0

    def load(self):
        if self.loaded:
            return
        print(f"[adv] loading ControlNet {CONTROLNET_ID} ...")
        t0 = time.time()
        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_ID, torch_dtype=DTYPE, use_safetensors=False
        )
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            MODEL_ID, controlnet=controlnet, torch_dtype=DTYPE,
            variant="fp16", safety_checker=None,
        ).to(DEVICE)
        self.pipe.set_progress_bar_config(disable=True)
        print(f"[adv] loaded in {time.time()-t0:.1f}s; warming up...")
        self._warmup()
        self.loaded = True
        print("[adv] ready.")

    def _warmup(self):
        dummy = Image.new("RGB", (SIZE, SIZE), (128, 128, 128))
        ctrl = self._canny(dummy, 100, 200)
        for _ in range(3):
            self._infer(dummy, ctrl, "a photo", 0.5, 2, 0.8, True)
        torch.cuda.synchronize()

    @staticmethod
    def _prep(frame: np.ndarray) -> Image.Image:
        img = Image.fromarray(frame)
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        return img.resize((SIZE, SIZE), Image.BILINEAR)

    @staticmethod
    def _canny(pil_img: Image.Image, lo: int, hi: int) -> Image.Image:
        edges = cv2.Canny(np.array(pil_img), lo, hi)
        return Image.fromarray(np.stack([edges] * 3, axis=-1))

    def _infer(self, pil_img, ctrl_img, prompt, strength, steps, ctrl_scale, fixed_seed, negative=None):
        generator = torch.Generator(DEVICE).manual_seed(1234) if fixed_seed else None
        out = self.pipe(
            prompt=prompt or "a photo",
            negative_prompt=negative or NEG_PROMPT,
            image=pil_img,
            control_image=ctrl_img,
            num_inference_steps=int(steps),
            strength=float(strength),
            guidance_scale=0.0,
            controlnet_conditioning_scale=float(ctrl_scale),
            generator=generator,
        ).images[0]
        return np.array(out)

    def reset_temporal(self):
        """Clear smoothing history (call on mode switch / big prompt change)."""
        self._smooth_prev = None

    def transform(self, frame, prompt, strength, steps, stabilize,
                  ctrl_scale, smoothing, negative=None, keep_bg=False):
        """Thread-safe. Returns (output_rgb_np, fps_text). Drops frame if busy.
        keep_bg: transform only the person, paste the original background back."""
        from segmenter import valid_frame
        if not valid_frame(frame):
            return self._last_out, self._fps_text()
        if not self._lock.acquire(blocking=False):
            return self._last_out, self._fps_text()
        try:
            pil = self._prep(frame)
            ctrl = self._canny(pil, 100, 200)
            out = self._infer(pil, ctrl, prompt, strength, steps, ctrl_scale, stabilize, negative=negative)

            # temporal smoothing on the generated content (before compositing bg)
            a = float(smoothing)
            if a > 0.0 and self._smooth_prev is not None:
                out = (a * self._smooth_prev.astype(np.float32)
                       + (1.0 - a) * out.astype(np.float32)).astype(np.uint8)
            self._smooth_prev = out

            # keep the real background crisp: composite person over original
            display = out
            if keep_bg:
                from segmenter import segmenter, composite_person
                segmenter.load()
                m = segmenter.mask(pil)
                display = composite_person(out, np.asarray(pil), m)

            self._tick_fps()
            self._last_out = display
            return display, self._fps_text()
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
            return "warming up..." if self.loaded else "not loaded"
        return f"{self._fps_ema:4.1f} fps   ({self._frames_done} frames)  [Advanced]"
