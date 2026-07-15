import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Kill ALL train.py processes
print("Killing all train.py...")
c.exec_command("pkill -9 -f train.py 2>/dev/null; true", timeout=5)
time.sleep(5)

# Reinstall torch
print("Reinstalling torch (may take time)...")
PIP = "/home/jiaqigu/hidnet_env/bin/pip"
stdin, stdout, stderr = c.exec_command(
    f"{PIP} install torch==2.5.1 --force-reinstall --no-deps 2>&1 | tail -5", timeout=120)
out = stdout.read().decode(errors="replace")
err = stderr.read().decode(errors="replace")
print(out[-200:] if len(out) > 200 else out)
print(err[-200:] if len(err) > 200 else err)

# Also reinstall torchvision, triton
print("Reinstalling torchvision, triton...")
c.exec_command(f"{PIP} install torchvision --force-reinstall --no-deps -q 2>&1 | tail -3", timeout=60)

# Verify torch
PY = "/home/jiaqigu/hidnet_env/bin/python"
stdin, stdout, stderr = c.exec_command(
    f"{PY} -c 'import torch; print(torch.__version__, torch.cuda.is_available())' 2>&1", timeout=15)
ver = stdout.read().decode().strip()
err_ver = stderr.read().decode().strip()
print(f"Torch: {ver} {err_ver[:200] if err_ver else ''}")

# Check free GPUs now
stdin, stdout, stderr = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader", timeout=10)
print("\nGPU status:")
for line in stdout.read().decode().strip().split("\n"):
    idx, used, total = line.replace("MiB","").replace(",","").split()
    free = int(total.strip()) - int(used.strip())
    print(f"  GPU{idx.strip()}: {used.strip()}/{total.strip()} MiB free={free}MiB")
c.close()
