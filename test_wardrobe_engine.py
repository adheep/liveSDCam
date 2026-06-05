"""Smoke-test WardrobeEngine: live transform + HQ capture."""
import numpy as np
from PIL import Image
from engine_wardrobe import WardrobeEngine

eng = WardrobeEngine()
eng.load()
frame = np.array(Image.open("test_before.png").convert("RGB"))

# live path
for i in range(6):
    out, fps = eng.transform(frame, "wearing an elegant tailored business suit and tie",
                             4, True, 0.3, negative=None)
    assert out is not None and out.shape == (512, 512, 3)
print("live OK:", out.shape, "| fps:", fps)
Image.fromarray(out).save("wardrobe_live_out.png")

# capture path (HQ)
cap = eng.capture(frame, "wearing a traditional white kurta, indian ethnic wear", steps=10)
assert cap is not None and cap.shape == (512, 512, 3)
Image.fromarray(cap).save("wardrobe_capture_out.png")
print("capture OK:", cap.shape)

# guard: empty frame should not crash
bad, _ = eng.transform(np.zeros((0, 0, 3), dtype=np.uint8), "suit", 4, True, 0.0)
print("empty-frame guard OK (returned last):", bad is not None)
print("=== WARDROBE ENGINE OK ===")
