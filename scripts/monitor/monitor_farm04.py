import paramiko, time, json, re, os
from datetime import datetime

HOST = "10.176.60.71"
REMOTE = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen"
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor_state.json")
INTERVAL = 600
TIMEOUT = 15


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def ssh_exec(c, cmd):
    stdin, o, e = c.exec_command(cmd, timeout=TIMEOUT)
    return o.read().decode(errors="replace").strip()


def parse_log_line(line):
    step_match = re.search(r'S(\d+)', line)
    loss_match = re.search(r'loss=([\d.]+(?:[eE][+-]?\d+)?)', line)
    eta_match = re.search(r'eta=([\d.]+[hm]?)', line)
    step = int(step_match.group(1)) if step_match else None
    loss = loss_match.group(1) if loss_match else None
    eta = eta_match.group(1) if eta_match else None
    return step, loss, eta


def check():
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(HOST, username="jiaqigu", password="lijia7272", timeout=TIMEOUT)

        log = ssh_exec(c, f"ls -t {REMOTE}/train_v3_*.log 2>/dev/null | head -1")
        gpu_raw = ssh_exec(c, "nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader")
        train_raw = ssh_exec(c, f"tail -3 {log} 2>/dev/null") if log else ""
        n_ckpt = ssh_exec(c, f"ls -lt {REMOTE}/step_*.pt 2>/dev/null | wc -l") if log else "0"
        n_proc = ssh_exec(c, "ps aux | grep jiaqigu | grep 'scripts/train.py' | grep -v grep | wc -l")
        c.close()
    except Exception as e:
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] [FAIL] SSH error: {e}")
        return

    gpu2 = None
    for l in gpu_raw.split("\n"):
        if l.startswith("2,"):
            p = [x.strip().replace(" MiB", "").replace(" %", "") for x in l.split(",")]
            gpu2 = {"mem": int(p[1]), "mem_mb": int(p[1]), "util": p[2], "temp": p[3]}
            break

    n_proc = n_proc.strip() or "0"
    n_ckpt = n_ckpt.strip() or "0"

    step = loss = eta = None
    for line in reversed(train_raw.split("\n")):
        s, l, e = parse_log_line(line)
        if s is not None: step = s
        if l is not None: loss = l
        if e is not None: eta = e
        if step is not None and loss is not None and eta is not None:
            break

    now = datetime.now()
    now_str = now.strftime("%H:%M:%S")
    state = load_state()
    warnings = []

    if n_proc == "0":
        warnings.append("process dead")

    if step is not None and state.get("last_step") is not None:
        if step == state["last_step"]:
            last_time = datetime.fromisoformat(state.get("last_time", now.isoformat()))
            stuck_min = (now - last_time).total_seconds() / 60
            if stuck_min >= 30:
                warnings.append(f"step stuck {int(stuck_min)}min")
    else:
        pass

    if step is not None:
        if step != state.get("last_step"):
            state["last_step"] = step
            state["last_time"] = now.isoformat()
    save_state(state)

    parts = [f"[{now_str}]"]
    if step is not None:
        parts.append(f"step={step}")
    if loss is not None:
        parts.append(f"loss={loss}")
    if eta is not None:
        parts.append(f"eta={eta}")
    if gpu2:
        parts.append(f"GPU2:mem={gpu2['mem']}MB/util={gpu2['util']}%/temp={gpu2['temp']}C")
    parts.append(f"ckpt={n_ckpt}")

    tags = []
    if warnings:
        tags.append("WARN:" + ",".join(warnings))
    else:
        tags.append("OK")

    print(" ".join(parts) + " [" + "|".join(tags) + "]", flush=True)


if __name__ == "__main__":
    print(f"Monitoring farm04 GPU2 every {INTERVAL}s. Ctrl+C to stop.", flush=True)
    while True:
        try:
            check()
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [ERR] {e}", flush=True)
        time.sleep(INTERVAL)
