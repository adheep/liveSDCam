"""Probe available webcams across backends/indices on Windows."""
import cv2

backends = [
    ("MSMF", cv2.CAP_MSMF),
    ("DSHOW", cv2.CAP_DSHOW),
    ("ANY", cv2.CAP_ANY),
]

print("Scanning camera indices 0-3 across backends...\n")
found = []
for bname, bid in backends:
    for idx in range(4):
        cap = cv2.VideoCapture(idx, bid)
        opened = cap.isOpened()
        ok, frame = (False, None)
        if opened:
            for _ in range(3):
                ok, frame = cap.read()
        cap.release()
        if opened and ok and frame is not None:
            h, w = frame.shape[:2]
            print(f"  [OK]   {bname:6} index {idx}  ->  {w}x{h}")
            found.append((bname, bid, idx, w, h))
        elif opened:
            print(f"  [open-but-no-frame] {bname:6} index {idx}")

print()
if found:
    print(f"WORKING CAMERA(S): {[(f[0], f[2]) for f in found]}")
else:
    print("NO working camera found. Likely causes:")
    print("  - Camera in use by another app (Teams/Zoom/Camera app)")
    print("  - Windows privacy: Settings > Privacy > Camera > 'Let desktop apps access camera'")
    print("  - Laptop camera disabled / external cam unplugged")
