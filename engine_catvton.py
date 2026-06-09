"""LiveSD CatVTON (capture) engine: faithful garment try-on.

Wraps the cloned CatVTON pipeline (third_party/CatVTON). Feeds our own segformer
garment mask (no DensePose). One-shot, high quality, ~15s/image -> Capture only.

To keep VRAM free for the other modes, the heavy modules live on CPU when idle
and are moved to the GPU only during a capture.
"""
import os
import sys
import threading
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

_CATVTON_DIR = os.path.join(os.path.dirname(__file__), "third_party", "CatVTON")
if _CATVTON_DIR not in sys.path:
    sys.path.insert(0, _CATVTON_DIR)

DEVICE = "cuda"
DTYPE = torch.bfloat16
BASE_CKPT = "booksforcharlie/stable-diffusion-inpainting"
ATTN_CKPT = "zhengchong/CatVTON"
PARSER_ID = "mattmdjaga/segformer_b2_clothes"
GARMENT_CLASSES = (4, 5, 6, 7, 8, 17)
H, W = 1024, 768


class CatVTONEngine:
    def __init__(self):
        self.pipe = None
        self.seg_proc = None
        self.seg_model = None
        self.loaded = False
        self._on_gpu = False
        self._lock = threading.Lock()

    def load(self):
        if self.loaded:
            return
        from model.pipeline import CatVTONPipeline
        print("[catvton] loading segformer + CatVTON pipeline ...")
        t0 = time.time()
        self.seg_proc = SegformerImageProcessor.from_pretrained(PARSER_ID)
        self.seg_model = AutoModelForSemanticSegmentation.from_pretrained(PARSER_ID).to(DEVICE).eval()
        self.pipe = CatVTONPipeline(
            base_ckpt=BASE_CKPT, attn_ckpt=ATTN_CKPT, attn_ckpt_version="mix",
            weight_dtype=DTYPE, device=DEVICE, skip_safety_check=True,
        )
        self._to_cpu()  # park on CPU until a capture is requested
        self.loaded = True
        print(f"[catvton] ready in {time.time()-t0:.1f}s (parked on CPU).")

    def _to_gpu(self):
        if not self._on_gpu:
            self.pipe.unet.to(DEVICE)
            self.pipe.vae.to(DEVICE)
            self._on_gpu = True

    def _to_cpu(self):
        self.pipe.unet.to("cpu")
        self.pipe.vae.to("cpu")
        self._on_gpu = False
        torch.cuda.empty_cache()

    @staticmethod
    def _prep_square(frame: np.ndarray, size=512) -> Image.Image:
        img = Image.fromarray(frame)
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        return img.resize((size, size), Image.BILINEAR)

    @torch.no_grad()
    def _garment_mask(self, pil_img, dilate=15) -> Image.Image:
        inputs = self.seg_proc(images=pil_img, return_tensors="pt").to(DEVICE)
        logits = self.seg_model(**inputs).logits
        up = F.interpolate(logits, size=pil_img.size[::-1], mode="bilinear", align_corners=False)
        labels = up.argmax(1)[0].cpu().numpy()
        m = np.isin(labels, GARMENT_CLASSES).astype(np.uint8) * 255
        m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8), iterations=1)
        return Image.fromarray(m)

    def capture(self, frame, garment, steps=30, guidance=2.5):
        """Faithful try-on of `garment` onto the current frame. Blocks (~15s)."""
        from segmenter import valid_frame
        if not valid_frame(frame) or not valid_frame(garment):
            return None
        with self._lock:
            person = self._prep_square(frame, 512)
            garment_pil = Image.fromarray(garment).convert("RGB")
            mask = self._garment_mask(person)             # same size as person (512)
            self._to_gpu()
            try:
                result = self.pipe(
                    image=person, condition_image=garment_pil, mask=mask,
                    num_inference_steps=int(steps), guidance_scale=float(guidance),
                    height=H, width=W,
                    generator=torch.Generator(DEVICE).manual_seed(42),
                )
            finally:
                self._to_cpu()
            result = result[0] if isinstance(result, list) else result
            return np.array(result.convert("RGB"))
