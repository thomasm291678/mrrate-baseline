import paramiko
import re
import sys
import time
import os
from datetime import datetime

HOST = "10.176.60.71"
USER = "jiaqigu"
PWD = "lijia7272"
LOG_DIR = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen"
PROJECT_DIR = "/home/jiaqigu/mrrate_hidnet"
PYTHON_BIN = "/home/jiaqigu/hidnet_env/bin/python"
CHECK_INTERVAL = 30 * 60  # 30 minutes

RECOVERY_CMD = (
    f"cd {PROJECT_DIR} && "
    f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    f"CUDA_VISIBLE_DEVICES=6 "
    f"nohup {PYTHON_BIN} -u scripts/train.py "
    f"--data_root /mnt/nas1/disk07/public/mr_data/MR-RATE "
    f"--v1_ckpt outputs/report_gen/best_model.pt "
    f"--qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct "
    f"--batch_size 5 --ga_steps 1 --epochs 5 "
    f"--num_workers 4 "
    f"--lr 1e-4 --cnn_lr 1e-5 --grid 2 "
    f"--vit_dim 512 --vit_heads 8 --vit_depth 2 "
    f"--use_amp --log_dir outputs/report_gen "
    f"--save_interval 2000 --log_interval 10 "
    f"--resume outputs/report_gen/epoch_*.pt "
    f"> outputs/report_gen/train_recovery_$(date +%Y%m%d_%H%M%S).log 2>&1 &"
)


def ssh_connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PWD, timeout=20)
    return ssh


def run_cmd(ssh, cmd, timeout=30):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    return stdout.read().decode("utf-8", errors="replace").strip()


def check_status(ssh):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'=' * 56}")
    print(f"  Farm01 Monitor  |  {now}")
    print(f"{'=' * 56}")

    # 1. Check python process
    proc_count = run_cmd(ssh, "pgrep -f train.py | wc -l")
    proc_alive = proc_count.strip() != "0" and proc_count.strip() != ""
    proc_detail = run_cmd(ssh, "ps -u jiaqigu | grep python")
    print(f"  Process     : {'ALIVE' if proc_alive else 'DEAD'} (train.py x{proc_count.strip()})")
    if proc_detail:
        for line in proc_detail.split("\n")[:3]:
            print(f"    {line.strip()[:100]}")

    # 2. Check GPU6
    gpu_out = run_cmd(ssh,
        "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu "
        "--format=csv,noheader | grep '^6'")
    print(f"  GPU6        : {gpu_out if gpu_out else 'NO DATA'}")

    # 3. Check latest log
    log_file = run_cmd(ssh, f"ls -t {LOG_DIR}/train_*.log 2>/dev/null | head -1")
    log_tail = ""
    if log_file:
        log_tail = run_cmd(ssh, f"tail -15 {log_file}")
        print(f"  Log ({os.path.basename(log_file)}):")
        for line in log_tail.split("\n")[-8:]:
            print(f"    {line.strip()[:120]}")

    # 4. Parse stats
    epoch = "?"
    step = "?"
    loss = "?"
    eta = "?"

    all_logs = log_tail
    if all_logs:
        em = re.findall(r'\bE(\d+)\b', all_logs)
        sm = re.findall(r'\bS(\d+)\b', all_logs)
        lm = re.findall(r'\bloss[=:]?\s*([\d.]+)', all_logs, re.IGNORECASE)
        etam = re.findall(r'\beta[=:]?\s*(\S+)', all_logs, re.IGNORECASE)
        epoch = em[-1] if em else "?"
        step = sm[-1] if sm else "?"
        loss = lm[-1] if lm else "?"
        eta = etam[-1] if etam else "?"

    print(f"{'=' * 56}")
    print(f"  Epoch: {epoch} | Step: {step} | Loss: {loss} | ETA: {eta}")
    print(f"{'=' * 56}")

    # Parse GPU info for memory/temp
    gpu_mem = "?"
    gpu_temp = "?"
    if gpu_out:
        parts = [p.strip() for p in gpu_out.split(",")]
        if len(parts) >= 5:
            gpu_mem = parts[1]
            gpu_temp = parts[4]

    return {
        "proc_alive": proc_alive,
        "proc_count": proc_count,
        "gpu_out": gpu_out,
        "gpu_mem": gpu_mem,
        "gpu_temp": gpu_temp,
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "eta": eta,
        "log_tail": log_tail,
    }


def do_recovery(ssh, status):
    print("\n" + "!" * 56)
    print("  CRASH DETECTED - Attempting auto-recovery...")
    print("!" * 56)

    # Kill zombies
    run_cmd(ssh, "pkill -9 -f train.py 2>/dev/null; true")
    time.sleep(2)

    # GPU6 cleanup
    stdout = run_cmd(ssh, "fuser -v /dev/nvidia6 2>/dev/null || echo 'no_fuser'")
    if "no_fuser" not in stdout and stdout.strip():
        run_cmd(ssh, "fuser -k /dev/nvidia6 2>/dev/null; true")
        time.sleep(2)

    # Check GPU6 is free
    gpu_check = run_cmd(ssh,
        "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep '^6'")
    print(f"  GPU6 pre-start: {gpu_check}")

    # Verify checkpoint exists for resume
    ckpt_check = run_cmd(ssh, f"ls {LOG_DIR}/epoch_*.pt 2>/dev/null | tail -3")
    print(f"  Checkpoints available: {ckpt_check[:200] if ckpt_check else 'NONE'}")

    # Start recovery training
    print(f"  Launching recovery training...")
    run_cmd(ssh, RECOVERY_CMD)
    time.sleep(10)

    # Verify
    proc_check = run_cmd(ssh, "ps -u jiaqigu | grep python")
    print(f"  Post-recovery python: {'OK' if 'python' in proc_check else 'FAILED'}")
    print("!" * 56)

    return "python" in proc_check.lower()


def main():
    print(f"Farm01 V3 MR-RATE Monitor starting")
    print(f"Target: {USER}@{HOST}")
    print(f"Interval: {CHECK_INTERVAL // 60} minutes")
    print(f"Log dir: {LOG_DIR}")
    print()

    cycle = 0
    while True:
        cycle += 1
        ssh = None
        try:
            ssh = ssh_connect()
            status = check_status(ssh)

            if not status["proc_alive"]:
                success = do_recovery(ssh, status)
                if not success:
                    print("  WARNING: Recovery may have failed - check manually!")
            else:
                print(f"  Status: OK | GPU Mem: {status['gpu_mem']} | Temp: {status['gpu_temp']}°C")

            ssh.close()
        except Exception as e:
            print(f"  ERROR: {e}")
            try:
                if ssh:
                    ssh.close()
            except:
                pass

        print(f"\n  Next check at {datetime.now().strftime('%H:%M:%S')} (+{CHECK_INTERVAL // 60}min)")
        sys.stdout.flush()
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
