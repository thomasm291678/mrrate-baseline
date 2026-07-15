#!/usr/bin/env python3
import subprocess, time, os, sys, json
from pathlib import Path
from datetime import datetime

PROJ = Path("/home/jiaqigu/mrrate_hidnet")
LOG_DIR = PROJ / "outputs/report_gen"
MON_LOG = LOG_DIR / "watchdog.log"
CHECK_INTERVAL = 300  # 5 minutes

PYTHON = "/home/jiaqigu/hidnet_env/bin/python"
TRAIN_SCRIPT = PROJ / "scripts/train.py"
DATA = "/mnt/nas1/disk07/public/mr_data/MR-RATE"
V1_CKPT = PROJ / "outputs/report_gen/best_model.pt"
QWEN = "/mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct"


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(MON_LOG, "a") as f:
        f.write(line + "\n")


def run(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.stderr.strip(), r.returncode
    except subprocess.TimeoutExpired:
        return "", "timeout", 1


def find_training_pid():
    out, _, _ = run("ps -u jiaqigu | grep 'python.*train.py' | grep -v grep | awk '{print $1}'")
    if out:
        return out.split("\n")[0]
    return None


def get_latest_log():
    logs = sorted(LOG_DIR.glob("train_*.log"))
    return logs[-1] if logs else None


def get_last_checkpoint():
    ckpts = sorted(LOG_DIR.glob("epoch_*.pt"))
    return ckpts[-1] if ckpts else None


def check_status():
    report = {}

    # GPU
    out, _, _ = run("nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader | grep '^6'")
    report["gpu"] = out

    # Process
    pid = find_training_pid()
    report["pid"] = pid

    # Latest log
    logfile = get_latest_log()
    if logfile:
        out, _, _ = run(f"tail -10 {logfile}")
        report["log_tail"] = out

    return report


def try_recover():
    ckpt = get_last_checkpoint()
    if not ckpt:
        log("RECOVER: no checkpoint found, cannot recover!")
        return False

    log(f"RECOVER: checkpoint={ckpt.name}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    recover_log = LOG_DIR / f"train_recover_{ts}.log"

    cmd = (
        f"cd {PROJ} && "
        f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        f"CUDA_VISIBLE_DEVICES=6 nohup {PYTHON} -u scripts/train.py "
        f"--data_root {DATA} "
        f"--v1_ckpt {V1_CKPT} "
        f"--qwen_path {QWEN} "
        f"--batch_size 5 --ga_steps 1 --epochs 5 --num_workers 4 "
        f"--lr 1e-4 --cnn_lr 1e-5 --grid 2 "
        f"--vit_dim 512 --vit_heads 8 --vit_depth 2 "
        f"--use_amp --log_dir outputs/report_gen "
        f"--save_interval 2000 --log_interval 10 "
        f"--resume {ckpt} "
        f"> {recover_log} 2>&1 &"
    )

    subprocess.run(cmd, shell=True)
    log(f"RECOVER: launched, log={recover_log.name}")

    time.sleep(15)
    pid = find_training_pid()
    if pid:
        log(f"RECOVER: new PID={pid}")
        return True
    else:
        log("RECOVER: failed to start!")
        return False


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 50)
    log("WATCHDOG DAEMON STARTED")
    log(f"Interval: {CHECK_INTERVAL}s  Project: {PROJ}")
    log("=" * 50)

    while True:
        report = check_status()

        pid = report["pid"]
        gpu = report["gpu"]
        log_tail = report.get("log_tail", "")

        if pid:
            # Extract key info from log tail
            loss_line = ""
            for line in log_tail.split("\n"):
                if "loss=" in line and "eta=" in line:
                    loss_line = line.strip()
            log(f"ALIVE PID={pid} GPU=[{gpu}]  {loss_line}")
        else:
            log("DEAD! No training process found. Attempting recovery...")
            time.sleep(30)
            try_recover()

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
