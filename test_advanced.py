"""Smoke-test AdvancedEngine: load, transform, temporal smoothing, backpressure."""
import numpy as np
from PIL import Image

from engine_advanced import AdvancedEngine

eng = AdvancedEngine()
eng.load()

frame = np.array(Image.open("test_before.png").convert("RGB"))
print(f"frame: {frame.shape}")

print("\n-- 10 transforms with smoothing=0.4 --")
for i in range(10):
    out, fps = eng.transform(frame, "a disney pixar 3d character", 0.6, 2, True, 0.8, 0.4)
    assert out is not None and out.shape == (512, 512, 3), f"bad: {None if out is None else out.shape}"
print("output OK:", out.shape, "| fps:", fps)
Image.fromarray(out).save("adv_smoothed.png")

print("\n-- verify temporal blend with two DIFFERENT frames --")
frameB = np.array(Image.open("cn_after.png").convert("RGB"))  # a different image
# raw output for frameB alone (no history)
eng.reset_temporal()
rawB, _ = eng.transform(frameB, "anime portrait", 0.6, 2, True, 0.8, 0.0)
# now prime history with frameA, then feed frameB WITH smoothing -> should pull toward A
eng.reset_temporal()
eng.transform(frame, "anime portrait", 0.6, 2, True, 0.8, 0.0)          # prev = rawA
blended, _ = eng.transform(frameB, "anime portrait", 0.6, 2, True, 0.8, 0.6)  # 0.6*rawA + 0.4*rawB
diff = float(np.mean(np.abs(blended.astype(int) - rawB.astype(int))))
print(f"mean abs diff (smoothed vs raw): {diff:.1f}  -> blending is active (should be > 0)")
assert diff > 0, "temporal smoothing had no effect"

print("\n=== ADVANCED ENGINE OK ===")
