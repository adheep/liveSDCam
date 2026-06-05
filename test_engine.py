"""Smoke-test the Engine end-to-end with real webcam frames + backpressure."""
import threading
import time

import cv2
import numpy as np

from engine import Engine

eng = Engine()
eng.load()

# grab a real frame
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
for _ in range(8):
    cap.read(); time.sleep(0.03)
ok, frame = cap.read()
cap.release()
assert ok, "webcam read failed"
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
print(f"frame: {frame.shape}")

# run 10 sequential transforms (simulates the stream loop)
print("\n-- sequential throughput --")
for i in range(10):
    out, fps = eng.transform(frame, "a disney pixar 3d character", 0.6, 2, True)
    assert out is not None and out.shape == (512, 512, 3), f"bad output: {None if out is None else out.shape}"
print("output shape OK:", out.shape, "| fps:", fps)

# test backpressure: hammer transform() from many threads, ensure no crash and frames drop
print("\n-- backpressure (8 threads, single-flight) --")
results = []
def worker():
    o, f = eng.transform(frame, "anime portrait", 0.6, 2, True)
    results.append(o is not None)
threads = [threading.Thread(target=worker) for _ in range(8)]
for t in threads: t.start()
for t in threads: t.join()
print(f"all {len(results)} calls returned a frame (dropped ones reuse last): {all(results)}")

print("\n=== ENGINE OK ===")
