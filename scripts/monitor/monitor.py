import paramiko
import time
import re
import json
import os
from datetime import datetime

HOST = "10.176.60.71"
USER = "jiaqigu"
PASS = "lijia7272"
REMOTE = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen"
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".monitor_state.json")
INTERVAL = 600

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"last_step": None, "last_step_ts": None}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def parse_log_line(line):
    step = loss = eta = None
    m_step = re.search(r"S(\d+)", line) or re.search(r"[Ss]tep\s*[:=]?\s*(\d+)", line) or re.search(r"(\d+)/\d+", line)
    if m_step:
        step = int(m_step.group(1))
    m_loss = re.search(r"loss=([\d.]+(?:e[+-]?\d+)?)", line) or re.search(r"[Ll]oss\s*[:=]?\s*([\d.]+(?:e[+-]?\d+)?)", line)
    if m_loss:
        loss = m_loss.group(1)
    m_eta = re.search(r"eta=(\S+)", line) or re.search(r"ETA\s*[:=]?\s*(\S+(?:\s*\S+)?)", line)
    if m_eta:
        eta = m_eta.group(1).strip().rstrip(",")
    return step, loss, eta

def parse_gpu2(gpu_line):
    parts = [x.strip().replace(" MiB", "").replace(" %", "") for x in gpu_line.split(",")]
    return parts[1], parts[2], parts[3]

def run_check():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS, timeout=15)

    _, o, _ = c.exec_command("nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader", timeout=10)
    gpu2_line = None
    for line in o.read().decode().strip().split("\n"):
        if line.startswith("2,"):
            gpu2_line = line
            break

    _, o, _ = c.exec_command(f"ls -t {REMOTE}/train_v3_*.log 2>/dev/null | head -1", timeout=10)
    log_path = o.read().decode().strip()

    _, o, _ = c.exec_command(f"tail -3 {log_path} 2>/dev/null", timeout=10)
    train_lines = o.read().decode(errors="replace").strip().split("\n")

    _, o, _ = c.exec_command(f"ls -lt {REMOTE}/step_*.pt 2>/dev/null | wc -l", timeout=10)
    n_ckpt = o.read().decode().strip()

    _, o, _ = c.exec_command("ps aux | grep 'scripts/train.py' | grep -v grep | wc -l", timeout=10)
    n_proc = o.read().decode().strip()

    c.close()

    now = datetime.now()
    now_ts = now.timestamp()
    state = load_state()

    proc_alive = int(n_proc) > 0
    gpu_mem, gpu_util, gpu_temp = ("?", "?", "?") if not gpu2_line else parse_gpu2(gpu2_line)

    step_str = "?"
    loss_str = "?"
    eta_str = "?"
    for line in reversed(train_lines):
        s, l, e = parse_log_line(line)
        if s is not None:
            step_str = str(s)
            if l:
                loss_str = l
            if e:
                eta_str = e
            break

    warn = ""
    if not proc_alive:
        warn = " [WARN:进程不存在]"
    elif step_str != "?" and state["last_step"] is not None and state["last_step_ts"] is not None:
        current_step = int(step_str)
        if current_step == state["last_step"] and (now_ts - state["last_step_ts"]) > 1800:
            stuck_min = int((now_ts - state["last_step_ts"]) / 60)
            warn = f" [WARN:Step卡住{stuck_min}分钟]"
        elif current_step > state["last_step"]:
            state["last_step"] = current_step
            state["last_step_ts"] = now_ts
        elif current_step < state["last_step"]:
            state["last_step"] = current_step
            state["last_step_ts"] = now_ts
    elif step_str != "?":
        state["last_step"] = int(step_str)
        state["last_step_ts"] = now_ts

    if gpu2_line and gpu_util == "0" and proc_alive:
        warn = warn + " [WARN:GPU利用率=0]"

    save_state(state)

    ts = now.strftime("%H:%M:%S")
    gpu_info = f"GPU2 | Mem={gpu_mem}/49152MB | Util={gpu_util}% | Temp={gpu_temp}°C"
    train_info = f"Step={step_str} | Loss={loss_str} | ETA={eta_str}"
    misc_info = f"Proc={n_proc} | CKPT={n_ckpt}"

    status = "OK" if not warn and proc_alive else ""
    print(f"[{ts}] {gpu_info} | {train_info} | {misc_info}{warn}")


if __name__ == "__main__":
    print(f"Starting monitor every {INTERVAL // 60}min. Press Ctrl+C to stop.\n")
    while True:
        try:
            run_check()
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [ERROR] {e}")
        time.sleep(INTERVAL)
