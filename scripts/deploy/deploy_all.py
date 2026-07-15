#!/usr/bin/env python3
"""
一体化部署脚本 — 从 farm02 备份到 NAS + 部署到 farm01 + 启动训练
运行位置: farm02 (10.154.32.115)
功能:
  1. 将所有文件备份到 NAS: /mnt/nas1/disk07/public/qi/
  2. 将代码部署到 farm01: /home/jiaqigu/mrrate_hidnet/
  3. 启动 farm01 GPU6 V3 训练
"""

import paramiko
import os
import subprocess
import time
import sys

FARM01_HOST = "10.176.60.71"
FARM01_USER = "jiaqigu"
FARM01_PWD = "lijia7272"
FARM01_REMOTE = "/home/jiaqigu/mrrate_hidnet"

NAS_ROOT = "/mnt/nas1/disk07/public/qi"
SRC = "/home/jiaqigu/mrrate_hidnet"


def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] {result.stderr[:200]}")
    return result.stdout.strip()


def step(msg):
    print(f"\n[{'='*5}] {msg}", flush=True)


# =============================================
# Phase 1: NAS Backup
# =============================================
def phase1_nas_backup():
    step("Phase 1: NAS Backup to /mnt/nas1/disk07/public/qi/")

    if not os.path.isdir(NAS_ROOT):
        print("ERROR: NAS not mounted!")
        print("Try: mount -t cifs //10.154.32.108/shared /mnt/nas1/disk07 -o username=guest")
        sys.exit(1)

    nas_dirs = ["weights", "code", "checkpoints", "logs"]
    for d in nas_dirs:
        os.makedirs(f"{NAS_ROOT}/{d}", exist_ok=True)

    # ---- copy model weights ----
    for fname in ["best_model.pt", "last_model.pt"]:
        src = f"{SRC}/outputs/report_gen/{fname}"
        dst = f"{NAS_ROOT}/weights/{fname}"
        if os.path.exists(src):
            sz = os.path.getsize(src)
            print(f"  weights/{fname}: {sz/1024**3:.1f}GB")
            run(f"rsync -ah --progress {src} {dst} 2>&1 | tail -1")
        else:
            print(f"  SKIP {fname}: not found")

    # ---- copy code files ----
    code_files = [
        (f"{SRC}/src/encoder.py", "encoder.py"),
        (f"{SRC}/src/densenet3d.py", "densenet3d.py"),
        (f"{SRC}/server_code/mrrate_dataset.py", "mrrate_dataset.py"),
        (f"{SRC}/scripts/train.py", "train.py"),
        (f"{SRC}/run.sh", "run.sh"),
        (f"{SRC}/watchdog.sh", "watchdog.sh"),
    ]
    for src, fname in code_files:
        dst = f"{NAS_ROOT}/code/{fname}"
        if os.path.exists(src):
            run(f"cp {src} {dst}")
            print(f"  code/{fname}")
        else:
            print(f"  SKIP code/{fname}: not found")

    # ---- copy checkpoints ----
    ckpt_dir = f"{SRC}/checkpoints"
    if os.path.isdir(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            src = f"{ckpt_dir}/{f}"
            dst = f"{NAS_ROOT}/checkpoints/{f}"
            run(f"cp {src} {dst}")
            print(f"  checkpoints/{f}")

    # ---- copy logs ----
    for log_dir in [f"{SRC}/logs", f"{SRC}/outputs/report_gen"]:
        if not os.path.isdir(log_dir):
            continue
        for f in os.listdir(log_dir):
            if f.endswith(('.log', '.txt')):
                src = f"{log_dir}/{f}"
                dst = f"{NAS_ROOT}/logs/{f}"
                run(f"cp {src} {dst}")
                print(f"  logs/{f}")

    # ---- README to root ----
    if not os.path.exists(f"{NAS_ROOT}/README.md"):
        readme_content = open(f"{NAS_ROOT}/code/README.md", "r").read() if os.path.exists(f"{NAS_ROOT}/code/README.md") else "See code/README.md"
        with open(f"{NAS_ROOT}/README.md", "w") as f:
            f.write(readme_content)
        print("  README.md -> root")

    print("  Phase 1: NAS backup DONE")
    run(f"du -sh {NAS_ROOT}/*/")
    return True


# =============================================
# Phase 2: Deploy to farm01
# =============================================
def phase2_deploy_farm01():
    step("Phase 2: Deploy to farm01 (10.176.60.71)")

    print("  Connecting to farm01...")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(FARM01_HOST, username=FARM01_USER, password=FARM01_PWD, timeout=30)

    # Create dirs
    c.exec_command(f"mkdir -p {FARM01_REMOTE}/{{src,scripts,outputs/report_gen,checkpoints}}")

    sftp = c.open_sftp()

    files_to_deploy = [
        ("encoder.py", f"{FARM01_REMOTE}/src/encoder.py"),
        ("densenet3d.py", f"{FARM01_REMOTE}/src/densenet3d.py"),
        ("mrrate_dataset.py", f"{FARM01_REMOTE}/server_code/mrrate_dataset.py"),
        ("train.py", f"{FARM01_REMOTE}/scripts/train.py"),
        ("run.sh", f"{FARM01_REMOTE}/run.sh"),
        ("watchdog.sh", f"{FARM01_REMOTE}/watchdog.sh"),
    ]

    for local_name, remote_path in files_to_deploy:
        local_paths = [
            os.path.join("C:", "\\Users\\HP\\Documents\\5555", local_name),
            os.path.join(os.path.dirname(__file__), local_name),
            f"{SRC}/src/{local_name}",
            f"{SRC}/scripts/{local_name}",
            f"{SRC}/{local_name}",
        ]
        uploaded = False
        for lp in local_paths:
            if os.path.exists(lp):
                print(f"  Upload: {local_name} ({os.path.getsize(lp)/1024:.0f}KB)")
                sftp.put(lp, remote_path)
                uploaded = True
                break
        if not uploaded:
            print(f"  SKIP {local_name}: not found locally or on farm02")

    sftp.close()

    # Chmod run/watchdog
    c.exec_command(f"chmod +x {FARM01_REMOTE}/run.sh {FARM01_REMOTE}/watchdog.sh")

    # Check env
    stdin, stdout, stderr = c.exec_command(
        f"ls {FARM01_REMOTE}/../hidnet_env/bin/python && echo ENV_OK || echo NO_ENV")
    env_status = stdout.read().decode().strip()
    if "NO_ENV" in env_status:
        print("  WARNING: hidnet_env not found on farm01! Need to transfer env first.")
        print("  Run nas_backup.py first to transfer environment.")
    else:
        print(f"  Env: OK")

    # Check GPU6
    stdin, stdout, stderr = c.exec_command(
        "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | grep '^6'")
    print(f"  GPU6: {stdout.read().decode().strip()}")

    c.close()
    print("  Phase 2: Deploy DONE")
    return True


# =============================================
# Phase 3: Start Training
# =============================================
def phase3_start_training():
    step("Phase 3: Start V3 Training on farm01 GPU6")

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(FARM01_HOST, username=FARM01_USER, password=FARM01_PWD, timeout=30)

    # Kill any existing training on GPU6
    c.exec_command("pkill -f 'CUDA_VISIBLE_DEVICES=6.*train.py' 2>/dev/null || true")
    time.sleep(2)

    # Launch training via watchdog (auto-restart on crash)
    cmd = (
        f"cd {FARM01_REMOTE} && "
        f"nohup bash watchdog.sh > watchdog_out.log 2>&1 &"
    )
    c.exec_command(cmd)
    print("  Watchdog started (auto-restart enabled)")

    # Wait a moment then check
    time.sleep(10)
    stdin, stdout, stderr = c.exec_command(
        "ps aux | grep -E 'train.py|watchdog' | grep -v grep")
    procs = stdout.read().decode().strip()
    print(f"  Running processes:\n{procs}")

    c.close()
    print("  Phase 3: Training Started")


# =============================================
# Main
# =============================================
def main():
    print("=" * 60)
    print("MR-RATE: NAS Backup + Farm01 Deploy + Train")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    phase1_nas_backup()
    phase2_deploy_farm01()

    ans = input("\nStart V3 training on farm01 GPU6? [y/N]: ").strip().lower()
    if ans == 'y':
        phase3_start_training()
    else:
        print("Skipped training start. Run phase3_start_training() manually.")

    print("\nAll done!")


if __name__ == "__main__":
    main()
