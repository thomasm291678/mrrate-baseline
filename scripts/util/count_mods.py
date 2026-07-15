import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

log = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v5_all_encoder_20260713_193843.log"

# Check forward_contrastive code: mods = ["t1", "t2", "flair"]
s, o, e = c.exec_command(f"grep -i 'mods\\|t1\\|t2\\|flair' /home/jiaqigu/mrrate_hidnet/scripts/train_v5.py | head -10", timeout=10)
print("=== Code ===")
print(o.read().decode())

# Count how many modalities exist per batch 
s, o, e = c.exec_command(
    f"cd /home/jiaqigu/mrrate_hidnet && "
    f"CUDA_VISIBLE_DEVICES=3 /home/jiaqigu/hidnet_env/bin/python -c \""
    "from mrrate_dataset import MRRateDataset; "
    "import numpy as np; "
    "ds = MRRateDataset('/mnt/nas1/disk07/public/mr_data/MR-RATE', 'train', augment=False); "
    "n1, n2, n3 = 0, 0, 0; "
    "for i in range(min(1000, len(ds))): "
    "  s = ds[i]; "
    "  n1 += int(s['has_t1']); n2 += int(s['has_t2']); n3 += int(s['has_flair']); "
    "print(f'T1: {n1}, T2: {n2}, Flair: {n3} / 1000'); "
    "c = 0; "
    "for i in range(min(1000, len(ds))): "
    "  s = ds[i]; "
    "  c += int(s['has_t1']) + int(s['has_t2']) + int(s['has_flair']) >= 2; "
    "print(f'Samples with >=2 mods: {c}/1000'); "
    "\" 2>&1",
    timeout=120)
print("\n=== Stats ===")
print(o.read().decode())

c.close()
