"""Benchmark engine.transform throughput (Lite). Run before/after optimization."""
import sys, time
import numpy as np
from PIL import Image
from engine import Engine

frame = np.array(Image.open("test_before.png").convert("RGB"))
eng = Engine()
eng.load()

# warmup (also triggers torch.compile if enabled)
for _ in range(5):
    eng.transform(frame, "a disney pixar character", 0.6, 2, True)

ts = []
for _ in range(25):
    t = time.time()
    out, _ = eng.transform(frame, "a disney pixar character", 0.6, 2, True)
    ts.append(time.time() - t)
ts.sort()
med = ts[len(ts) // 2]
print(f"\nLite img2img (512, 2 steps): {med*1000:6.1f} ms  ->  {1/med:4.1f} fps   (best {1/ts[0]:.1f})")
