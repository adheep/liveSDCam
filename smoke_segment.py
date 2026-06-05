"""Smoke-test person segmentation + background compositing.
Cuts the transformed person (test_after.png) and pastes it over the ORIGINAL
background (test_before.png). Saves mask + composite, measures speed.
"""
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.segmentation import (
    deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights,
)

DEVICE, SIZE = "cuda", 512
PERSON_CLASS = 15  # VOC label index for 'person'
SEG_RES = 384      # run segmentation at this res for speed, upsample mask to SIZE

weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
model = deeplabv3_mobilenet_v3_large(weights=weights).eval().to(DEVICE)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


@torch.no_grad()
def person_mask(pil_img, feather=11):
    arr = np.asarray(pil_img.convert("RGB").resize((SEG_RES, SEG_RES))).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    t = (t - MEAN) / STD
    logits = model(t)["out"][0]                      # (21, H, W)
    person = (logits.argmax(0) == PERSON_CLASS).float()[None, None]
    person = F.interpolate(person, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
    m = person[0, 0].cpu().numpy()
    if feather > 0:
        m = cv2.GaussianBlur(m, (feather, feather), 0)
    return np.clip(m, 0, 1)


src = Image.open("test_before.png").convert("RGB").resize((SIZE, SIZE))   # original (bg source)
transformed = Image.open("test_after.png").convert("RGB").resize((SIZE, SIZE))  # "stylized person"

# warmup + timing
person_mask(src)
ts = []
for _ in range(15):
    t = time.time(); m = person_mask(src); torch.cuda.synchronize(); ts.append(time.time() - t)
ts.sort(); seg_ms = ts[len(ts) // 2] * 1000

src_np = np.asarray(src).astype(np.float32)
tr_np = np.asarray(transformed).astype(np.float32)
m3 = m[..., None]
composite = (m3 * tr_np + (1 - m3) * src_np).astype(np.uint8)

mask_vis = Image.fromarray((m * 255).astype(np.uint8)).convert("RGB")
combo = Image.new("RGB", (SIZE * 4 + 30, SIZE), "white")
for i, im in enumerate([src, transformed, mask_vis, Image.fromarray(composite)]):
    combo.paste(im, (i * (SIZE + 10), 0))
combo.save("segment_combo.png")
print(f"segmentation: {seg_ms:.0f} ms/frame")
print("saved segment_combo.png  [source | transformed-full | mask | composite(person+orig bg)]")
