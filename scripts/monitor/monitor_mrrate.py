import paramiko
import time
import re
from datetime import datetime

HOST = "10.176.60.71"
REMOTE = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen"
INTERVAL = 600

last_step = None
last_step_time = None


def ssh():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username="jiaqigu", password="lijia7272", timeout=15)
    return c


def run(cmd, client):
    stdin, o, e = client.exec_command(cmd, timeout=10)
    return o.read().decode(errors="replace").strip()


def parse_log(lines):
    for line in reversed(lines.split("\n")):
        step_m = re.search(r'[Ss](\d{5,6})', line)
        if not step_m:
            step_m = re.search(r'step[:\s=]*(\d+)', line, re.IGNORECASE)
        loss_m = re.search(r'loss[=:\s]*([\d.]+)', line, re.IGNORECASE)
        eta_m = re.search(r'eta[=:\s]*(\d+\.?\d*\s*h|\d+h\d+m|\d+:\d+:\d+|\d+\s*min)', line, re.IGNORECASE)
        if step_m:
            step = step_m.group(1)
            loss = loss_m.group(1) if loss_m else "?"
            eta = eta_m.group(1) if eta_m else "?"
            return step, loss, eta
    return "?", "?", "?"


def check():
    global last_step, last_step_time

    try:
        c = ssh()
    except Exception as e:
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] [WARN] SSH 连接失败: {e}", flush=True)
        return

    try:
        gpu_raw = run(
            "nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader",
            c,
        )
        gpu2 = "?"
        for line in gpu_raw.split("\n"):
            if line.startswith("2,"):
                p = [x.strip().replace(" MiB", "").replace(" %", "") for x in line.split(",")]
                gpu2 = f"mem={p[1]}/49152MB util={p[2]}% temp={p[3]}°C"
                break

        log_path = run(f"ls -t {REMOTE}/train_v3_*.log 2>/dev/null | head -1", c)

        train_raw = run(f"tail -3 {log_path} 2>/dev/null", c)

        n_ckpt = run(f"ls {REMOTE}/step_*.pt 2>/dev/null | wc -l", c)

        n_proc = run("ps -u jiaqigu | grep train.py | grep -v grep | wc -l", c)
        if n_proc == "0":
            n_proc2 = run(r"ps -u jiaqigu | grep -E 'train_v3|train\.py' | grep -v grep | wc -l", c)
            if n_proc2 != "0":
                n_proc = n_proc2
        if n_proc == "0":
            log_mtime = run(f"stat -c %Y {log_path} 2>/dev/null", c)
            if log_mtime:
                try:
                    age = time.time() - int(log_mtime)
                    if age < 300:
                        n_proc = "1"
                except ValueError:
                    pass

        c.close()
    except Exception as e:
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] [WARN] 命令执行失败: {e}", flush=True)
        try:
            c.close()
        except Exception:
            pass
        return

    step, loss, eta = parse_log(train_raw)
    now = datetime.now()
    ts = now.strftime("%H:%M:%S")
    warn = ""

    if n_proc == "0" or n_proc == "":
        warn = " [WARN] 训练进程不存在"

    if not warn and step != "?":
        if last_step is not None and last_step == step and last_step_time is not None:
            if (now - last_step_time).total_seconds() > 1800:
                warn = f" [WARN] Step {step} 卡住超过30分钟"
        if last_step != step:
            last_step = step
            last_step_time = now

    gpu_short = gpu2 if gpu2 != "?" else "GPU:?"
    print(f"[{ts}] Step:{step} | Loss:{loss} | ETA:{eta} | {gpu_short} | CKPT:{n_ckpt}{warn}", flush=True)


if __name__ == "__main__":
    print(f"farm04 GPU2 MR-RATE 监控启动，每{INTERVAL // 60}分钟检查一次...", flush=True)
    while True:
        check()
        time.sleep(INTERVAL)
