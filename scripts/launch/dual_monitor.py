import paramiko
from datetime import datetime

LOG_PATH = r"C:\Users\HP\Documents\5555\dual_monitor.log"

def log(msg):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log(f"=== MR-RATE Dual Monitor [{now_str}] ===")

SERVERS = [
    {
        "name": "farm04:GPU6",
        "host": "10.176.60.71",
        "gpu_index": "6",
        "log_glob": "/home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_gpu6_*.log",
        "idle_mem_mb": 20000,
    },
    {
        "name": "farm05:GPU0",
        "host": "10.176.60.72",
        "gpu_index": "0",
        "log_glob": "/home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_farm05_*.log",
        "idle_mem_mb": 30000,
    },
]

for srv in SERVERS:
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(srv["host"], username="jiaqigu", password="lijia7272", timeout=15)

        _, o, _ = c.exec_command(
            f"tail -30 $(ls -t {srv['log_glob']} 2>/dev/null | head -1) 2>/dev/null",
            timeout=10,
        )
        raw = o.read().decode(errors="replace").strip()

        step_line = ""
        for line in reversed(raw.split("\n")):
            if "S0" in line and ("loss=" in line or "Loss" in line):
                step_line = line.strip()
                break
        if not step_line:
            for line in reversed(raw.split("\n")):
                if "S0" in line:
                    step_line = line.strip()
                    break
        if not step_line:
            step_line = raw.split("\n")[-1].strip() if raw else "?"

        _, o, _ = c.exec_command(
            "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader",
            timeout=10,
        )
        gpu_lines = o.read().decode().strip().split("\n")
        gpu = ""
        for gl in gpu_lines:
            if gl.startswith(srv["gpu_index"] + ","):
                gpu = gl
                break

        parts = [x.strip() for x in gpu.split(",")] if gpu else []
        mem_str = parts[1].replace(" MiB", "") if len(parts) > 1 else "?"
        util_str = parts[2].replace(" %", "") if len(parts) > 2 else "?"

        try:
            mem = int(mem_str)
            util = int(util_str)
        except ValueError:
            mem, util = 0, 0

        if util > 10:
            tag = "OK"
        elif mem > srv["idle_mem_mb"]:
            tag = "IDLE"
        else:
            tag = "DEAD"

        log(f"  {srv['name']} [{tag}] VRAM={mem}MB UT={util}% | {step_line}")
        c.close()

    except Exception as ex:
        log(f"  {srv['name']} [DOWN] {ex}")
