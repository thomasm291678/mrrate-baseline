import paramiko
import time
import os
from datetime import datetime

LOG_DIR = r"C:\Users\HP\Documents\5555"
LOCAL_LOG = os.path.join(LOG_DIR, "training_status.log")
INTERVAL = 300  # 5 minutes

HOST = "10.176.60.71"
USER = "jiaqigu"
PWD = "lijia7272"
REMOTE = "/home/jiaqigu/mrrate_hidnet"


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOCAL_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fetch():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PWD, timeout=20)

    report = []

    cmd1 = "tail -3 $(ls -t " + REMOTE + "/outputs/report_gen/train_*.log | head -1) 2>/dev/null"
    stdin, stdout, stderr = c.exec_command(cmd1, timeout=15)
    out = stdout.read().decode(errors="replace").strip()
    report.append("TRAIN: " + (out.split("\n")[-1] if out else "no log"))

    cmd2 = "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader 2>/dev/null | grep '^6'"
    stdin, stdout, stderr = c.exec_command(cmd2, timeout=15)
    out = stdout.read().decode(errors="replace").strip()
    report.append("GPU6: " + (out or "not found"))

    cmd3 = "ps -u jiaqigu 2>/dev/null | grep train.py | grep -v grep | wc -l"
    stdin, stdout, stderr = c.exec_command(cmd3, timeout=15)
    out = stdout.read().decode(errors="replace").strip()
    report.append("PROCS: " + (out or "0"))

    cmd4 = "tail -1 " + REMOTE + "/outputs/report_gen/monitor.log 2>/dev/null"
    stdin, stdout, stderr = c.exec_command(cmd4, timeout=15)
    out = stdout.read().decode(errors="replace").strip()
    report.append("MON: " + (out or "no server mon"))

    cmd5 = "ls -t " + REMOTE + "/outputs/report_gen/epoch_*.pt 2>/dev/null | head -1"
    stdin, stdout, stderr = c.exec_command(cmd5, timeout=15)
    out = stdout.read().decode(errors="replace").strip()
    report.append("CKPT: " + (out or "none yet"))

    c.close()

    log(" | ".join(report))


def main():
    log("=" * 60)
    log("LOCAL MONITOR STARTED (5min interval)")
    log("Log: " + LOCAL_LOG)
    log("=" * 60)

    while True:
        try:
            fetch()
        except Exception as e:
            log(f"ERROR: {e}")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
