import paramiko

# Upload smoke test script
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

script = '''
import sys, time
sys.path.insert(0, "/home/jiaqigu/mrrate_hidnet")
from encoder_v5 import MRRCNN
from mrrate_dataset import MRRateDataset
import torch, torch.nn as nn

dev = torch.device("cuda:0")

# 1. Load model
print("Loading model...")
model = MRRCNN(in_chans=1, base_ch=32, grid=2).to(dev)
n = sum(p.numel() for p in model.parameters())
print(f"  params: {n:,}")

# 2. Load dataset
print("Loading dataset...")
ds = MRRateDataset(
    root="/mnt/nas1/disk07/public/mr_data/MR-RATE",
    split="train", augment=False,
)
print(f"  samples: {len(ds)}")

# 3. One batch
loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0, shuffle=True)
batch = next(iter(loader))
x = batch["t1"].to(dev)
x = x.reshape(x.shape[0], 1, 128, 128, 128)
print(f"  input: {tuple(x.shape)}")

# 4. Forward
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
out = model(x)
torch.cuda.synchronize()
t_forward = time.time() - t0
mem = torch.cuda.max_memory_allocated() / 1e9
print(f"  output: {tuple(out.shape)}, forward: {t_forward:.2f}s, peak mem: {mem:.1f}GB")

# 5. Backward
target = torch.randn_like(out)
t0 = time.time()
loss = nn.functional.mse_loss(out, target)
loss.backward()
torch.cuda.synchronize()
t_backward = time.time() - t0
mem_total = torch.cuda.max_memory_allocated() / 1e9
print(f"  loss: {loss.item():.4f}, backward: {t_backward:.2f}s, total peak: {mem_total:.1f}GB")

# 6. Speed test
print("\\nSpeed test (10 iterations)...")
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
for _ in range(10):
    out = model(x)
    loss = nn.functional.mse_loss(out, target)
    loss.backward()
torch.cuda.synchronize()
t_total = time.time() - t0
print(f"  10 iters: {t_total:.1f}s, per iter: {t_total/10:.2f}s")
print(f"  peak mem: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

print("\\nSMOKE TEST PASSED")
'''

sftp = c.open_sftp()
from io import BytesIO
sftp.putfo(BytesIO(script.encode()), "/tmp/smoke_v5.py")
sftp.close()

import time as tm
start = tm.time()

s, o, e = c.exec_command(
    "cd /home/jiaqigu/mrrate_hidnet && "
    "CUDA_VISIBLE_DEVICES=3 "
    "/home/jiaqigu/hidnet_env/bin/python /tmp/smoke_v5.py 2>&1",
    timeout=120)
out = o.read().decode().strip()
err = e.read().decode().strip()
print(out)
if err:
    print("STDERR:", err)
print(f"\\nTotal: {tm.time()-start:.0f}s")

c.close()
