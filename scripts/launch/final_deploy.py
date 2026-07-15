import paramiko
import os
import time
import sys

FARM02 = {"host": "10.154.32.115", "user": "jiaqigu", "password": "lijia7272"}
FARM01 = {"host": "10.176.60.71", "user": "jiaqigu", "password": "lijia7272"}

LOCAL = r"C:\Users\HP\Documents\5555"
FARM02_REMOTE = "/home/jiaqigu/mrrate_hidnet"
FARM01_REMOTE = "/home/jiaqigu/mrrate_hidnet"
NAS_ROOT = "/mnt/nas1/disk07/public/qi"


def ssh_exec(client, cmd, timeout=120):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    return out, err


def step(msg):
    print(f"\n{'='*50}")
    print(f"  {msg}")
    print(f"{'='*50}", flush=True)


# ======== Phase 1: Upload code to farm02 ========
def phase1_upload_to_farm02():
    step("Phase 1: Upload adjusted code to farm02")

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(**FARM02, timeout=30)
    sftp = c.open_sftp()

    uploads = [
        ("encoder.py", f"{FARM02_REMOTE}/src/encoder.py"),
        ("train.py", f"{FARM02_REMOTE}/scripts/train.py"),
        ("run.sh", f"{FARM02_REMOTE}/run.sh"),
        ("watchdog.sh", f"{FARM02_REMOTE}/watchdog.sh"),
        ("server_code/densenet3d.py", f"{FARM02_REMOTE}/src/densenet3d.py"),
        ("server_code/mrrate_dataset.py", f"{FARM02_REMOTE}/server_code/mrrate_dataset.py"),
        ("README.md", f"{FARM02_REMOTE}/README.md"),
    ]

    for local_rel, remote in uploads:
        local_path = os.path.join(LOCAL, local_rel)
        if not os.path.exists(local_path):
            print(f"  SKIP: {local_rel} (not found)")
            continue
        sz = os.path.getsize(local_path)
        print(f"  Upload: {local_rel} ({sz/1024:.0f}KB) -> {remote}")
        try:
            sftp.put(local_path, remote)
        except Exception as e:
            print(f"  ERROR: {e}")

    sftp.close()

    # Upload NAS backup scripts
    sftp = c.open_sftp()
    for script in ["nas_backup.py", "deploy_all.py"]:
        local_path = os.path.join(LOCAL, script)
        remote_path = f"{FARM02_REMOTE}/{script}"
        if os.path.exists(local_path):
            sftp.put(local_path, remote_path)
            print(f"  Upload: {script} -> {remote_path}")
    sftp.close()

    c.exec_command(f"chmod +x {FARM02_REMOTE}/run.sh {FARM02_REMOTE}/watchdog.sh")
    print("  Phase 1: Done")
    c.close()
    return True


# ======== Phase 2: NAS backup from farm02 ========
def phase2_nas_backup():
    step("Phase 2: NAS backup from farm02")

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(**FARM02, timeout=30)

    # Check NAS
    out, err = ssh_exec(c, f"df -h {NAS_ROOT} 2>/dev/null && echo NAS_OK || echo NAS_NOT_MOUNTED")
    print(f"  NAS: {out.strip()[-80:]}")
    if "NAS_NOT_MOUNTED" in out:
        print("  ERROR: NAS not mounted on farm02!")
        c.close()
        return False

    # Create dirs
    ssh_exec(c, f"mkdir -p {NAS_ROOT}/weights {NAS_ROOT}/code {NAS_ROOT}/checkpoints {NAS_ROOT}/logs")

    sftp = c.open_sftp()

    # Copy model weights to NAS
    for fname in ["best_model.pt", "last_model.pt"]:
        src = f"{FARM02_REMOTE}/outputs/report_gen/{fname}"
        dst_nas = f"{NAS_ROOT}/weights/{fname}"
        try:
            sz = sftp.stat(src).st_size
            print(f"  weights/{fname}: {sz/1024**3:.1f}GB...", end=" ", flush=True)
            t0 = time.time()
            # Use cp on the server side (faster than SFTP transfer)
            ssh_exec(c, f"cp {src} {dst_nas}")
            elapsed = time.time() - t0
            print(f"done ({elapsed:.0f}s)")
        except Exception as e:
            print(f"  SKIP weights/{fname}: {e}")

    # Copy code files to NAS
    code_files = [
        ("src/encoder.py", "encoder.py"),
        ("src/densenet3d.py", "densenet3d.py"),
        ("server_code/mrrate_dataset.py", "mrrate_dataset.py"),
        ("scripts/train.py", "train.py"),
        ("run.sh", "run.sh"),
        ("watchdog.sh", "watchdog.sh"),
        ("README.md", "README.md"),
    ]
    for remote_rel, nas_name in code_files:
        src = f"{FARM02_REMOTE}/{remote_rel}"
        dst = f"{NAS_ROOT}/code/{nas_name}"
        try:
            sftp.stat(src)
            ssh_exec(c, f"cp {src} {dst}")
            print(f"  code/{nas_name} ✓")
        except:
            print(f"  code/{nas_name} ✗")

    # Copy checkpoints
    ssh_exec(c, f"cp -r {FARM02_REMOTE}/checkpoints/* {NAS_ROOT}/checkpoints/ 2>/dev/null")
    chk = ssh_exec(c, f"ls {NAS_ROOT}/checkpoints/ | wc -l")
    print(f"  checkpoints: {chk[0].strip()} files")

    # Copy logs
    for log_dir in ["logs", "outputs/report_gen"]:
        src_dir = f"{FARM02_REMOTE}/{log_dir}"
        try:
            files = sftp.listdir(src_dir)
            for f in files:
                if f.endswith(('.log', '.txt')):
                    ssh_exec(c, f"cp {src_dir}/{f} {NAS_ROOT}/logs/{f}")
            print(f"  logs from {log_dir} ✓")
        except:
            pass

    # Copy README to NAS root
    ssh_exec(c, f"cp {FARM02_REMOTE}/README.md {NAS_ROOT}/README.md 2>/dev/null")

    sftp.close()

    # Summary
    out, _ = ssh_exec(c, f"du -sh {NAS_ROOT}/*/ 2>/dev/null")
    print(f"\n  NAS summary:\n{out}")

    c.close()
    print("  Phase 2: NAS backup DONE")
    return True


# ======== Phase 3: Deploy to farm01 ========
def phase3_deploy_farm01():
    step("Phase 3: Deploy to farm01 (10.176.60.71)")

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(**FARM01, timeout=30)

    # Verify GPU6
    out, _ = ssh_exec(c, "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | grep '^6'")
    print(f"  GPU6: {out.strip()}")

    # Kill old processes
    ssh_exec(c, "pkill -f 'CUDA_VISIBLE_DEVICES=6.*train.py' 2>/dev/null || true")
    ssh_exec(c, "pkill -f 'watchdog.sh' 2>/dev/null || true")
    time.sleep(2)

    # Create directories
    ssh_exec(c, f"mkdir -p {FARM01_REMOTE}/{{src,scripts,outputs/report_gen,checkpoints}}")

    sftp = c.open_sftp()

    uploads = [
        ("encoder.py", f"{FARM01_REMOTE}/src/encoder.py"),
        ("train.py", f"{FARM01_REMOTE}/scripts/train.py"),
        ("run.sh", f"{FARM01_REMOTE}/run.sh"),
        ("watchdog.sh", f"{FARM01_REMOTE}/watchdog.sh"),
        ("server_code/densenet3d.py", f"{FARM01_REMOTE}/src/densenet3d.py"),
        ("server_code/mrrate_dataset.py", f"{FARM01_REMOTE}/server_code/mrrate_dataset.py"),
    ]

    for local_rel, remote in uploads:
        local_path = os.path.join(LOCAL, local_rel)
        if not os.path.exists(local_path):
            print(f"  SKIP: {local_rel}")
            continue
        print(f"  Upload: {local_rel} -> {remote}")
        sftp.put(local_path, remote)

    sftp.close()
    c.exec_command(f"chmod +x {FARM01_REMOTE}/run.sh {FARM01_REMOTE}/watchdog.sh")

    # Verify env
    out, _ = ssh_exec(c, f"test -f /home/jiaqigu/hidnet_env/bin/python && echo ENV_OK || echo NO_ENV")
    if "NO_ENV" in out:
        print("  WARNING: hidnet_env not on farm01! Need to transfer from farm02.")
        # Try SFTP from farm02 to farm01 of env tar
        print("  Transferring env from farm02 to farm01 via farm02 SSH...")
        c.close()

        # Connect to farm02 to push env to farm01
        c2 = paramiko.SSHClient()
        c2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c2.connect(**FARM02, timeout=30)

        # Create env tar on farm02
        out, _ = ssh_exec(c2, f"ls /tmp/farm01_env.tar.gz 2>/dev/null && echo TAR_EXISTS || echo NO_TAR")
        if "NO_TAR" in out:
            print("  Creating env tar on farm02 (this will take time)...")
            ssh_exec(c2, "tar czf /tmp/farm01_env.tar.gz --exclude=__pycache__ --exclude='*.pyc' -C /home/jiaqigu hidnet_env env_pkg", timeout=600)

        # Now push from farm02 to farm01
        print("  Pushing env from farm02 to farm01...")
        # Copy to farm01 via SFTP from farm02
        ssh_exec(c2, f"""
python3 -c "
import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('{FARM01["host"]}', username='{FARM01["user"]}', password='{FARM01["password"]}', timeout=60)
s = c.open_sftp()
s.put('/tmp/farm01_env.tar.gz', '/home/jiaqigu/farm01_env.tar.gz')
s.close()
c.exec_command('cd /home/jiaqigu && tar xzf farm01_env.tar.gz && rm -f farm01_env.tar.gz')
c.close()
print('ENV_TRANSFER_DONE')
"
""", timeout=1200)
        print("  Env transfer initiated")
        c2.close()
    else:
        print("  Env: OK")

    # Check if v1 weights on farm01
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(**FARM01, timeout=30)
    out, _ = ssh_exec(c, f"ls -lh {FARM01_REMOTE}/outputs/report_gen/best_model.pt 2>/dev/null || echo NO_MODEL")
    if "NO_MODEL" in out:
        print("  best_model.pt not on farm01, need to upload...")
        local_model = os.path.join(LOCAL, "weights", "best_model.pt")
        if os.path.exists(local_model):
            sz = os.path.getsize(local_model)
            print(f"  Uploading best_model.pt ({sz/1024**3:.1f}GB) to farm01...")
            sftp = c.open_sftp()
            sftp.put(local_model, f"{FARM01_REMOTE}/outputs/report_gen/best_model.pt")
            sftp.close()
            print("  best_model.pt uploaded")
        else:
            print(f"  Local best_model.pt not found at {local_model}")
    else:
        print(f"  Model: {out.strip()}")

    c.close()
    print("  Phase 3: Deploy DONE")
    return True


# ======== Phase 4: Start Training ========
def phase4_start_training():
    step("Phase 4: Start V3 Training on farm01 GPU6")

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(**FARM01, timeout=30)

    # Verify GPU6 free
    out, _ = ssh_exec(c, "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep '^6'")
    gpu_mem = out.split(",")[1].strip() if "," in out else "unknown"
    print(f"  GPU6 memory used: {gpu_mem}")

    mem_mb = int(gpu_mem.split()[0]) if "MiB" in gpu_mem else 0
    if mem_mb > 1000:
        print(f"  WARNING: GPU6 has {gpu_mem} in use. Killing old processes...")
        ssh_exec(c, "fuser -k /dev/nvidia6 2>/dev/null || true")
        time.sleep(3)

    # Start watchdog
    cmd = f"cd {FARM01_REMOTE} && nohup bash watchdog.sh > /dev/null 2>&1 &"
    ssh_exec(c, cmd)
    print("  Watchdog started")

    time.sleep(8)

    # Check training started
    out, _ = ssh_exec(c, "ps aux | grep -E 'train.py|python' | grep -v grep")
    print(f"  Processes:\n{out}")

    # Check latest log
    ssh_exec(c, f"sleep 15 && tail -20 {FARM01_REMOTE}/outputs/report_gen/train_*.log 2>/dev/null || echo 'No log yet'")
    out, _ = ssh_exec(c, f"ls -t {FARM01_REMOTE}/outputs/report_gen/train_*.log 2>/dev/null | head -1")
    latest_log = out.strip()
    if latest_log:
        out, _ = ssh_exec(c, f"tail -30 {latest_log}")
        print(f"  Latest log ({latest_log}):\n{out[-2000:]}")
    else:
        print("  No log file yet - training may still be initializing")

    c.close()
    print("  Phase 4: Training started")
    return True


# ======== Main ========
def main():
    print("=" * 60)
    print("  MR-RATE: Full Deploy Pipeline")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    phase1_upload_to_farm02()
    phase2_nas_backup()
    phase3_deploy_farm01()

    ans = input("\nStart V3 training on farm01 GPU6? [y/N]: ").strip().lower()
    if ans == 'y':
        phase4_start_training()
    else:
        print("Skipped. Run phase4_start_training() manually to start.")

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
