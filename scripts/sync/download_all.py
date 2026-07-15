import paramiko
import os
import time
import sys

HOST = "10.154.32.115"
USER = "jiaqigu"
PWD = "lijia7272"

LOCAL = r"C:\Users\HP\Documents\5555"
REMOTE = "/home/jiaqigu/mrrate_hidnet"

os.makedirs(os.path.join(LOCAL, "weights"), exist_ok=True)
os.makedirs(os.path.join(LOCAL, "training_logs"), exist_ok=True)
os.makedirs(os.path.join(LOCAL, "server_code"), exist_ok=True)
os.makedirs(os.path.join(LOCAL, "checkpoints"), exist_ok=True)

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PWD, timeout=30)
sftp = client.open_sftp()


def download(remote_path, local_path, label):
    try:
        sz = sftp.stat(remote_path).st_size
    except:
        print(f"  [SKIP] {label}: {remote_path} not found")
        return
    print(f"  {label}: {sz/1024**3:.2f}GB -> {local_path}")
    t0 = time.time()
    sftp.get(remote_path, local_path)
    elapsed = time.time() - t0
    speed = sz / elapsed / 1024**2
    print(f"  done in {elapsed:.0f}s ({speed:.1f} MB/s)")


def download_dir_files(remote_dir, local_dir, label, pattern=None):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
    try:
        files = sftp.listdir(remote_dir)
    except:
        print(f"  [SKIP] {label}: {remote_dir} not found")
        return
    for f in files:
        if pattern and pattern not in f:
            continue
        rp = f"{remote_dir}/{f}"
        lp = os.path.join(local_dir, f)
        try:
            sz = sftp.stat(rp).st_size
        except:
            continue
        print(f"  {label}/{f}: {sz/1024:.0f}KB -> {lp}")
        sftp.get(rp, lp)


print("=" * 50)
print("Downloading from farm02...")
print("=" * 50)

print("\n--- Model Weights ---")
download(f"{REMOTE}/outputs/report_gen/best_model.pt", os.path.join(LOCAL, "weights", "best_model.pt"), "best_model.pt (V1 final)")
download(f"{REMOTE}/outputs/report_gen/last_model.pt", os.path.join(LOCAL, "weights", "last_model.pt"), "last_model.pt (V1 last)")

print("\n--- Code Files ---")
download_dir_files(f"{REMOTE}/src", os.path.join(LOCAL, "server_code"), "src")
download_dir_files(f"{REMOTE}/scripts", os.path.join(LOCAL, "server_code"), "scripts")
download(f"{REMOTE}/run.sh", os.path.join(LOCAL, "server_code", "run.sh"), "run.sh")
download(f"{REMOTE}/watchdog.sh", os.path.join(LOCAL, "server_code", "watchdog.sh"), "watchdog.sh")

print("\n--- Training Logs ---")
log_dirs = [
    f"{REMOTE}/logs",
    f"{REMOTE}/outputs/report_gen",
]
for ld in log_dirs:
    try:
        files = sftp.listdir(ld)
        for f in files:
            if f.endswith(('.log', '.txt')):
                rp = f"{ld}/{f}"
                lp = os.path.join(LOCAL, "training_logs", f)
                try:
                    sz = sftp.stat(rp).st_size
                    print(f"  {f}: {sz/1024:.0f}KB")
                    sftp.get(rp, lp)
                except:
                    continue
    except:
        pass

print("\n--- Checkpoints ---")
try:
    ckpt_files = sftp.listdir(f"{REMOTE}/checkpoints")
    for f in ckpt_files:
        rp = f"{REMOTE}/checkpoints/{f}"
        lp = os.path.join(LOCAL, "checkpoints", f)
        try:
            sz = sftp.stat(rp).st_size
            if sz > 1024 * 1024:
                print(f"  {f}: {sz/1024**3:.2f}GB")
            else:
                print(f"  {f}: {sz/1024:.0f}KB")
            sftp.get(rp, lp)
        except:
            continue
except:
    print("  No checkpoints dir")

sftp.close()
client.close()
print("\nALL DOWNLOADS COMPLETE")
