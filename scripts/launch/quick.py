import paramiko, time, re, signal, sys, functools
from datetime import datetime

print = functools.partial(print, flush=True)

HOST = "10.176.60.71"
REMOTE = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen"
INTERVAL = 600
STUCK_THRESHOLD = 1800

last_step = None
last_step_time = None
running = True

def signal_handler(sig, frame):
    global running
    running = False
    print("\n[STOP] Monitoring stopped.")

signal.signal(signal.SIGINT, signal_handler)

def get_ssh():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username="jiaqigu", password="lijia7272", timeout=15)
    return c

def parse_log(text):
    result = {"step": None, "loss": None, "eta": None}
    for line in reversed(text.split("\n")):
        s = re.search(r'(?:step|Step|STEP)[\s:=]*(\d+)', line)
        if s: result["step"] = int(s.group(1))
        l = re.search(r'(?:loss|Loss|LOSS)[\s:=]*([\d.]+)', line)
        if l: result["loss"] = float(l.group(1))
        e = re.search(r'(?:eta|ETA)[\s:=]*([\dhms,\s]+?)(?:\s*[,|]|\s*$)', line)
        if e: result["eta"] = e.group(1).strip()
        if result["step"] is not None:
            break
    return result

def check():
    global last_step, last_step_time
    try:
        c = get_ssh()
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [WARN] SSH failed: {e}")
        return

    try:
        stdin, o, e = c.exec_command(f"ls -t {REMOTE}/train_v3_*.log 2>/dev/null | head -1", timeout=10)
        log_path = o.read().decode().strip()
        if not log_path:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [WARN] No training log found")
            c.close()
            return

        stdin, o, e = c.exec_command(
            "nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader", timeout=10)
        gpu_line = None
        for l in o.read().decode().strip().split("\n"):
            if l.startswith("2,"):
                gpu_line = l
                break

        stdin, o, e = c.exec_command(f"tail -3 {log_path} 2>/dev/null", timeout=10)
        train_text = o.read().decode(errors="replace").strip()

        stdin, o, e = c.exec_command(f"ls {REMOTE}/step_*.pt 2>/dev/null | wc -l", timeout=10)
        n_ckpt = o.read().decode().strip()

        stdin, o, e = c.exec_command("ps -u jiaqigu | grep train.py | grep -v grep | wc -l", timeout=10)
        n_proc = int(o.read().decode().strip())
        c.close()
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [WARN] Query error: {e}")
        try: c.close()
        except: pass
        return

    parsed = parse_log(train_text)

    now = datetime.now()
    ts = now.strftime('%H:%M:%S')

    gpu_str = "GPU:?"
    if gpu_line:
        p = [x.strip().replace(" MiB","").replace(" %","") for x in gpu_line.split(",")]
        gpu_str = f"VRAM={p[1]}MB UT={p[2]}% T={p[3]}C"

    step_str = f"S:{parsed['step']}" if parsed['step'] is not None else "S:?"
    loss_str = f"L:{parsed['loss']:.4f}" if parsed['loss'] is not None else "L:?"
    eta_str = f"ETA:{parsed['eta']}" if parsed['eta'] else "ETA:?"

    flags = []
    if n_proc == 0:
        flags.append("NO_PROC")
    else:
        if parsed['step'] is not None:
            if last_step is not None and parsed['step'] == last_step:
                elapsed = (now - last_step_time).total_seconds()
                if elapsed > STUCK_THRESHOLD:
                    flags.append(f"STUCK({int(elapsed//60)}m)")
            elif last_step is not None and parsed['step'] != last_step:
                pass
            last_step = parsed['step']
            last_step_time = now

    tag = " [WARN] " + ",".join(flags) if flags else " [OK]"

    print(f"[{ts}]{tag} {step_str} {loss_str} {eta_str} | {gpu_str} | CKPT:{n_ckpt} PROC:{n_proc}")

if __name__ == "__main__":
    print(f"=== MR-RATE Monitor: farm04 GPU2, every {INTERVAL//60}min ===")
    while running:
        check()
        for _ in range(INTERVAL):
            if not running:
                break
            time.sleep(1)
