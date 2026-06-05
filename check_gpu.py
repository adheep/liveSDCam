import torch

print(f"torch version      : {torch.__version__}")
print(f"CUDA available     : {torch.cuda.is_available()}")
print(f"CUDA (torch built) : {torch.version.cuda}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA NOT available - go/no-go FAILED")

dev = torch.device("cuda")
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU                : {name}")
print(f"Compute capability : sm_{cap[0]}{cap[1]}")
print(f"Total VRAM         : {total_gb:.1f} GB")

# Real fp16 matmul on the GPU to confirm Blackwell kernels actually run
a = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
b = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
torch.cuda.synchronize()
import time
t0 = time.time()
for _ in range(10):
    c = a @ b
torch.cuda.synchronize()
dt = (time.time() - t0) / 10
tflops = (2 * 4096**3) / dt / 1e12
print(f"fp16 matmul        : OK  ({dt*1000:.2f} ms/iter, ~{tflops:.0f} TFLOPS)")

# bf16 check (preferred for diffusion on Blackwell)
x = torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
print("bf16 matmul        : OK")
print("\n=== GO: GPU path works ===")
