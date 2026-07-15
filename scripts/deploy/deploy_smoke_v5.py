import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Upload all files
sftp = c.open_sftp()
sftp.put(r"C:\Users\HP\Documents\5555\encoder_v5.py", "/home/jiaqigu/mrrate_hidnet/encoder_v5.py")
sftp.put(r"C:\Users\HP\Documents\5555\train_v5.py", "/home/jiaqigu/mrrate_hidnet/scripts/train_v5.py")
sftp.put(r"C:\Users\HP\Documents\5555\server_code\mrrate_dataset.py", "/home/jiaqigu/mrrate_hidnet/mrrate_dataset.py")
sftp.close()
print("Uploaded encoder_v5.py, train_v5.py, mrrate_dataset.py")

smoke = '''
import sys, time
sys.path.insert(0, "/home/jiaqigu/mrrate_hidnet")
from encoder_v5 import ReportingModelV5, MRRCNN
from mrrate_dataset import MRRateDataset
import torch, torch.nn as nn

dev = torch.device("cuda:0")

print("=== Phase 1: Contrastive Learning ===")
ds = MRRateDataset("/mnt/nas1/disk07/public/mr_data/MR-RATE", "train", augment=False)
print(f"Dataset: {len(ds)} samples")

from scripts.train_v5 import collate_fn, forward_contrastive, contrastive_loss_fn
loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=True)
batch = next(iter(loader))
print(f"Batch: {batch['t1'].shape}, patients: {batch['patient_id'].tolist()}")

enc = ReportingModelV5(llm_dim=2048, grid=2, base_ch=32).to(dev)
for k, v in batch.items():
    if isinstance(v, torch.Tensor) and v.dtype in (torch.float32, torch.float64):
        batch[k] = v.to(dev)
    elif k == "patient_id":
        batch[k] = v.to(dev)

opt = torch.optim.AdamW(enc.parameters(), lr=1e-4)

t0 = time.time()
loss = forward_contrastive(enc, batch, dev)
if loss.requires_grad:
    loss.backward()
torch.cuda.synchronize()
print(f"  loss: {loss.item():.4f}, time: {time.time()-t0:.2f}s")
print(f"  mem: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

print("\\n=== Speed test (10 iters) ===")
torch.cuda.reset_peak_memory_stats()
opt.zero_grad()
t0 = time.time()
for _ in range(10):
    loss = forward_contrastive(enc, batch, dev)
    loss.backward()
    opt.step()
    opt.zero_grad()
torch.cuda.synchronize()
print(f"  10 iters: {time.time()-t0:.1f}s, per iter: {(time.time()-t0)/10:.2f}s")
print(f"  peak mem: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

print("\\nSMOKE TEST PASSED")
'''

from io import BytesIO
sftp = c.open_sftp()
sftp.putfo(BytesIO(smoke.encode()), "/tmp/smoke_v5_2.py")
sftp.close()

s, o, e = c.exec_command(
    "cd /home/jiaqigu/mrrate_hidnet && CUDA_VISIBLE_DEVICES=3 "
    "/home/jiaqigu/hidnet_env/bin/python /tmp/smoke_v5_2.py 2>&1",
    timeout=120)
print(o.read().decode().strip())
err = e.read().decode().strip()
if err:
    print("STDERR:", err)

c.close()
