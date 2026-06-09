"""Smoke-test TryOnEngine (live) + CatVTONEngine (capture) end-to-end."""
import numpy as np
from PIL import Image
from engine_tryon import TryOnEngine
from engine_catvton import CatVTONEngine

person = np.array(Image.open("test_before.png").convert("RGB"))
garment = np.array(Image.open("garment_suit.png").convert("RGB"))

print("--- TryOn (IP-Adapter live) ---")
te = TryOnEngine()
te.load()
for _ in range(5):
    out, fps = te.transform(person, garment, "a person wearing this outfit",
                            4, True, 0.3, 0.8, negative=None)
    assert out is not None and out.shape == (512, 512, 3)
print("live OK:", out.shape, "| fps:", fps)
Image.fromarray(out).save("tryon_live_out.png")
# no-garment guard
g0, f0 = te.transform(person, None, "x", 4, True, 0.0, 0.8)
print("no-garment guard OK (returns last):", g0 is not None or f0 is not None)

print("\n--- CatVTON (capture, on-demand GPU) ---")
ce = CatVTONEngine()
ce.load()
print("parked on GPU?", ce._on_gpu, "(should be False)")
cap = ce.capture(person, garment, steps=20)
assert cap is not None and cap.ndim == 3
print("capture OK:", cap.shape, "| back on CPU?", not ce._on_gpu)
Image.fromarray(cap).save("tryon_capture_out.png")
print("\n=== TRY-ON ENGINES OK ===")
