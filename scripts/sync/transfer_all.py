import paramiko, subprocess, os, time, sys

FARM01_HOST = "10.176.60.71"
FARM01_USER = "jiaqigu"
FARM01_PWD = "lijia7272"
SRC = "/home/jiaqigu"

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERR: {result.stderr[:200]}")
    return result.stdout.strip()

def step(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

step("Tar hidnet_env + env_pkg")
env_size = int(run("du -sb /home/jiaqigu/hidnet_env /home/jiaqigu/env_pkg | awk '{s+=$1}END{print s}'"))
print(f"  env size: {env_size/1024**3:.1f}GB")

step("Tar env...")
run(f"tar czf /tmp/farm01_env.tar.gz --exclude=__pycache__ --exclude='*.pyc' -C {SRC} hidnet_env env_pkg")
env_tar_sz = os.path.getsize("/tmp/farm01_env.tar.gz")
print(f"  env tar: {env_tar_sz/1024**3:.1f}GB")

step("Tar code...")
run(f"tar czf /tmp/farm01_code.tar.gz --exclude=outputs/report_gen/best_model.pt --exclude=outputs/report_gen/last_model.pt --exclude=outputs/pretrain_densenet/*.pt --exclude=__pycache__ --exclude='*.pyc' -C {SRC} mrrate_hidnet")
code_tar_sz = os.path.getsize("/tmp/farm01_code.tar.gz")
print(f"  code tar: {code_tar_sz/1024**3:.2f}GB")

step("Connecting to farm01...")
c1 = paramiko.SSHClient()
c1.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c1.connect(FARM01_HOST, username=FARM01_USER, password=FARM01_PWD, timeout=30)

c1.exec_command(f"mkdir -p {SRC}/mrrate_hidnet/outputs/report_gen")

step("Upload env tar (this will take time)...")
t0 = time.time()
sftp = c1.open_sftp()
sftp.put("/tmp/farm01_env.tar.gz", f"{SRC}/farm01_env.tar.gz")
elapsed = time.time() - t0
speed = env_tar_sz / elapsed / 1024**2
print(f"  env uploaded in {elapsed:.0f}s ({speed:.1f} MB/s)")

step("Upload code tar...")
t0 = time.time()
sftp.put("/tmp/farm01_code.tar.gz", f"{SRC}/farm01_code.tar.gz")
print(f"  code uploaded in {time.time()-t0:.0f}s")

step("Upload best_model.pt...")
model_path = f"{SRC}/mrrate_hidnet/outputs/report_gen/best_model.pt"
model_sz = os.path.getsize(model_path)
t0 = time.time()
sftp.put(model_path, model_path)
speed = model_sz / (time.time() - t0) / 1024**2
print(f"  model uploaded in {time.time()-t0:.0f}s ({speed:.1f} MB/s)")
sftp.close()

step("Extract on farm01...")
stdin, stdout, stderr = c1.exec_command(f"cd {SRC} && tar xzf {SRC}/farm01_env.tar.gz && rm -f {SRC}/farm01_env.tar.gz && tar xzf {SRC}/farm01_code.tar.gz && rm -f {SRC}/farm01_code.tar.gz", timeout=600)
out = stdout.read().decode()
err = stderr.read().decode()
if err:
    print(f"  extract err: {err[:200]}")
else:
    print("  extracted OK")

step("Verify...")
stdin, stdout, stderr = c1.exec_command(f"{SRC}/hidnet_env/bin/python -c 'import torch; print(torch.__version__)'")
print(f"  torch: {stdout.read().decode().strip()}")

stdin, stdout, stderr = c1.exec_command(f"ls -lh {SRC}/mrrate_hidnet/outputs/report_gen/best_model.pt {SRC}/mrrate_hidnet/src/densenet3d.py")
print(stdout.read().decode().strip())

c1.close()

run("rm -f /tmp/farm01_env.tar.gz /tmp/farm01_code.tar.gz")
step("ALL DONE!")
print("ALL_DONE_MARKER", flush=True)
