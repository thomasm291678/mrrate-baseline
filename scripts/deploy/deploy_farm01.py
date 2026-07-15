import paramiko
import os
import time

FARM01 = {"hostname": "10.176.60.71", "username": "jiaqigu", "password": "lijia7272"}
LOCAL = r"C:\Users\HP\Documents\5555"
REMOTE = "/home/jiaqigu/mrrate_hidnet"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(**FARM01, timeout=30)

# Kill old processes
print("Killing old processes...")
c.exec_command("pkill -f 'CUDA_VISIBLE_DEVICES=6.*train.py' 2>/dev/null; pkill -f 'watchdog.sh' 2>/dev/null; true")
time.sleep(2)

# Ensure directories
c.exec_command(f"mkdir -p {REMOTE}/server_code")

# Upload new code
sftp = c.open_sftp()

uploads = [
    ("encoder.py", f"{REMOTE}/src/encoder.py"),
    ("train.py", f"{REMOTE}/scripts/train.py"),
    ("run.sh", f"{REMOTE}/run.sh"),
    ("watchdog.sh", f"{REMOTE}/watchdog.sh"),
    ("server_code/densenet3d.py", f"{REMOTE}/src/densenet3d.py"),
    ("server_code/mrrate_dataset.py", f"{REMOTE}/server_code/mrrate_dataset.py"),
]

for local_rel, remote in uploads:
    local_path = os.path.join(LOCAL, local_rel)
    if not os.path.exists(local_path):
        print(f"  SKIP: {local_rel}")
        continue
    sz = os.path.getsize(local_path)
    print(f"  Upload: {local_rel} ({sz}B) -> {os.path.basename(remote)}")
    sftp.put(local_path, remote)

sftp.close()
c.exec_command(f"chmod +x {REMOTE}/run.sh {REMOTE}/watchdog.sh")
print("Code uploaded!")

# Verify GPU6
stdin, stdout, stderr = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep '^6'")
gpu6 = stdout.read().decode().strip()
print(f"GPU6: {gpu6}")

mem_mb = int(gpu6.split(",")[1].strip().split()[0]) if "MiB" in gpu6 else 0
if mem_mb > 1000:
    print("GPU6 busy, killing...")
    c.exec_command("fuser -k /dev/nvidia6 2>/dev/null; sleep 3; true")

# Verify env and model
c.exec_command(f"test -f /home/jiaqigu/hidnet_env/bin/python && echo ENV_OK || echo NO_ENV")
c.exec_command(f"test -f {REMOTE}/outputs/report_gen/best_model.pt && echo MODEL_OK || echo NO_MODEL")

# Start watchdog
print("\nStarting watchdog (auto-restart on crash)...")
c.exec_command(f"cd {REMOTE} && nohup bash watchdog.sh > /tmp/watchdog_out.log 2>&1 &")
print("Watchdog launched!")

# Wait and check 
time.sleep(15)

stdin, stdout, stderr = c.exec_command("ps aux | grep -E 'train.py|watchdog' | grep -v grep")
procs = stdout.read().decode().strip()
print(f"\nRunning:\n{procs}")

# Check log
stdin, stdout, stderr = c.exec_command(
    f"ls -t {REMOTE}/outputs/report_gen/train_*.log 2>/dev/null | head -1")
latest_log = stdout.read().decode().strip()
if latest_log:
    stdin, stdout, stderr = c.exec_command(f"tail -20 {latest_log}")
    print(f"\nLatest log ({latest_log}):")
    print(stdout.read().decode())
else:
    print("\nNo log yet, training may still be initializing...")
    stdin, stdout, stderr = c.exec_command(f"cat {REMOTE}/watchdog_out.log 2>/dev/null")
    wd = stdout.read().decode()
    if wd:
        print(f"Watchdog output:\n{wd[:500]}")

c.close()
print("\nDeploy complete! Training started on farm01 GPU6.")
