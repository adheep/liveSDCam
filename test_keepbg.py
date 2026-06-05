"""Verify Lite engine keep_bg path end-to-end (transform person, keep bg)."""
import numpy as np
from PIL import Image
from engine import Engine

eng = Engine()
eng.load()
frame = np.array(Image.open("test_before.png").convert("RGB"))

out_full, _ = eng.transform(frame, "a disney pixar 3d character", 0.6, 2, True, keep_bg=False)
out_bg, fps = eng.transform(frame, "a disney pixar 3d character", 0.6, 2, True, keep_bg=True)
assert out_bg.shape == (512, 512, 3)

# The two should differ (bg pixels restored to original in keep_bg version)
diff = float(np.mean(np.abs(out_full.astype(int) - out_bg.astype(int))))
print(f"keep_bg output shape: {out_bg.shape} | fps: {fps}")
print(f"mean abs diff full-vs-keepbg: {diff:.1f} (should be > 0)")
assert diff > 0
Image.fromarray(out_bg).save("keepbg_out.png")
print("saved keepbg_out.png\n=== KEEP_BG OK ===")
