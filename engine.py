"""LiveSD inference engine: SD-Turbo img2img with single-flight backpressure.

Keeps the pipeline warm and exposes a thread-safe transform() that drops
frames when the GPU is busy instead of queuing them (no lag buildup).
"""
import threading
import time

import numpy as np
import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import logging as dl

dl.set_verbosity_error()

DEVICE = "cuda"
DTYPE = torch.float16
MODEL_ID = "stabilityai/sd-turbo"
SIZE = 512
NEG_PROMPT = "blurry, distorted, deformed, ugly, low quality, extra limbs"


class Engine:
    def __init__(self):
        self.pipe = None
        self._lock = threading.Lock()      # single-flight: only one inference at a time
        self._last_out = None              # last successful output (returned when busy)
        self._fps_ema = None
        self._last_ts = None
        self._frames_done = 0

    def load(self):
        print(f"[engine] loading {MODEL_ID} ...")
        t0 = time.time()
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            MODEL_ID, torch_dtype=DTYPE, variant="fp16", safety_checker=None
        ).to(DEVICE)
        self.pipe.set_progress_bar_config(disable=True)
        from perf import optimize_pipe
        optimize_pipe(self.pipe, DTYPE, DEVICE, use_taesd=True, compile_unet=True, label=":lite")
        print(f"[engine] loaded in {time.time()-t0:.1f}s; warming up (compiling)...")
        self._warmup()
        print("[engine] ready.")

    def _warmup(self):
        dummy = Image.new("RGB", (SIZE, SIZE), (128, 128, 128))
        for _ in range(5):  # extra iters so torch.compile finishes before serving
            self._infer(dummy, "a photo", 0.5, 2, fixed_seed=True)
        torch.cuda.synchronize()

    @staticmethod
    def _prep(frame: np.ndarray) -> Image.Image:
        """np RGB frame -> center-cropped square SIZExSIZE PIL image."""
        img = Image.fromarray(frame)
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        return img.resize((SIZE, SIZE), Image.BILINEAR)

    def _infer(self, pil_img, prompt, strength, steps, fixed_seed, negative=None):
        # Fixed seed greatly reduces frame-to-frame flicker (temporal stability).
        generator = torch.Generator(DEVICE).manual_seed(1234) if fixed_seed else None
        out = self.pipe(
            prompt=prompt or "a photo",
            negative_prompt=negative or NEG_PROMPT,
            image=pil_img,
            num_inference_steps=int(steps),
            strength=float(strength),
            guidance_scale=0.0,
            generator=generator,
        ).images[0]
        return np.array(out)

    def transform(self, frame, prompt, strength, steps, stabilize, negative=None, keep_bg=False):
        """Thread-safe. Returns (output_rgb_np, fps_text).
        Drops the frame (returns last output) if an inference is already running.
        keep_bg: transform only the person, paste the original background back."""
        from segmenter import valid_frame
        if not valid_frame(frame):
            return self._last_out, self._fps_text()

        if not self._lock.acquire(blocking=False):
            # GPU busy with a previous frame -> skip this one, keep feed live
            return self._last_out, self._fps_text()
        try:
            pil = self._prep(frame)
            out = self._infer(pil, prompt, strength, steps, fixed_seed=stabilize, negative=negative)
            if keep_bg:
                from segmenter import segmenter, composite_person
                segmenter.load()
                m = segmenter.mask(pil)
                out = composite_person(out, np.asarray(pil), m)
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
            return "warming up..."
        return f"{self._fps_ema:4.1f} fps   ({self._frames_done} frames)"
