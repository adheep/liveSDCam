"""Shared person segmenter + background-compositing helper.

Lets Lite/Advanced transform ONLY the person and keep the real webcam
background. Uses torchvision DeepLabV3-MobileNet on the GPU (no extra deps).
Lazy-loaded the first time 'Keep background' is enabled.
"""
import threading

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights,
)

DEVICE = "cuda"
PERSON_CLASS = 15      # VOC 'person'
SEG_RES = 256          # segmentation input res (speed/quality balance)
SIZE = 512


class PersonSegmenter:
    def __init__(self):
        self.model = None
        self.loaded = False
        self._lock = threading.Lock()
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

    def load(self):
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            print("[seg] loading DeepLabV3-MobileNet person segmenter ...")
            w = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
            self.model = deeplabv3_mobilenet_v3_large(weights=w).eval().to(DEVICE)
            self.loaded = True
            print("[seg] ready.")

    @torch.no_grad()
    def mask(self, pil_img: Image.Image, feather: int = 11) -> np.ndarray:
        """Return a float [0,1] person mask at SIZE x SIZE (1 = person)."""
        arr = np.asarray(pil_img.convert("RGB").resize((SEG_RES, SEG_RES))).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        t = (t - self._mean) / self._std
        logits = self.model(t)["out"][0]
        person = (logits.argmax(0) == PERSON_CLASS).float()[None, None]
        person = F.interpolate(person, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        m = person[0, 0].cpu().numpy()
        if feather > 0:
            m = cv2.GaussianBlur(m, (feather, feather), 0)
        return np.clip(m, 0.0, 1.0)


def valid_frame(frame) -> bool:
    """Guard against empty/zero-size frames Gradio can emit on (dis)connect,
    which otherwise crash the VAE with a 0-element reshape."""
    return (frame is not None
            and getattr(frame, "ndim", 0) == 3
            and getattr(frame, "size", 0) > 0
            and frame.shape[0] > 0 and frame.shape[1] > 0)


def composite_person(transformed: np.ndarray, original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """person (from transformed) over original background, via soft mask."""
    m3 = mask[..., None]
    return (m3 * transformed.astype(np.float32)
            + (1.0 - m3) * original.astype(np.float32)).astype(np.uint8)


# module-level singleton shared by all engines
segmenter = PersonSegmenter()
