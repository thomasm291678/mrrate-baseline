import paramiko
import re
import time
from datetime import datetime, timedelta

HOST = "10.176.60.71"
REMOTE = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen"
USER = "jiaqigu"
PASS = "lijia7272"
INTERVAL = 600  # 10 minutes
STALL_MINUTES = 30

prev_step = None
prev_step_ts = None


def ssh_connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS, timeout=15)
    return c


def run_remote(c, cmd, timeout=10):
    stdin, o, e = c.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors="replace").strip()


def parse_log_line(line):
    step_match = re.search(r'(?:S(\d+)|[Ss]tep[\s:=]*(\d+))', line)
    loss_match = re.search(r'loss[\s:=]*([\d.]+(?:e[+-]?\d+)?)', line)
    eta_match = re.search(r'eta[\s:=]*([\d.]+[hm]?)', line)
    lr_match = re.search(r'lr[\s:=]*([\d.e+-]+)', line)
    step = int(step_match.group(1) or step_match.group(2)) if step_match else None
    loss = float(loss_match.group(1)) if loss_match else None
    eta = eta_match.group(1) if eta_match else None
    lr = float(lr_match.group(1)) if lr_match else None
    return step, loss, eta, lr


def check():
    global prev_step, prev_step_ts

    c = None
    gpu_str = "?"
    proc_str = "?"
    ckpt_str = "?"
    log_str = "?"
    status = "OK"

    try:
        c = ssh_connect()

        log_file = run_remote(c, f"ls -t {REMOTE}/train_v3_*.log 2>/dev/null | head -1")
        if not log_file:
            status = "[WARN] no log found"
        else:
            tail = run_remote(c, f"tail -3 {log_file} 2>/dev/null")
            lines = tail.split("\n") if tail else []

            step = loss = eta = lr = None
            for line in reversed(lines):
                s, l, e, lr_val = parse_log_line(line)
                if s is not None:
                    step = s
                if l is not None and loss is None:
                    loss = l
                if e is not None and eta is None:
                    eta = e
                if lr_val is not None and lr is None:
                    lr = lr_val
                if step is not None:
                    break

            if step is not None:
                now = datetime.now()
                if prev_step is not None and prev_step_ts is not None:
                    if step == prev_step:
                        stalled_min = (now - prev_step_ts).total_seconds() / 60
                        if stalled_min > STALL_MINUTES:
                            status = f"[WARN] step {step} stalled {stalled_min:.0f}m"
                    else:
                        prev_step = step
                        prev_step_ts = now
                else:
                    prev_step = step
                    prev_step_ts = now

                parts = [f"step={step}"]
                if loss is not None:
                    parts.append(f"loss={loss:.4f}")
                if eta is not None:
                    parts.append(f"eta={eta}")
                if lr is not None:
                    parts.append(f"lr={lr:.2e}")
                log_str = " ".join(parts)
            else:
                log_str = "no step info"

        gpu_info = run_remote(c, "nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader", timeout=10)
        gpu2 = [l for l in gpu_info.split("\n") if l.strip().startswith("2,")]
        if gpu2:
            p = [x.strip().replace(" MiB", "").replace(" %", "") for x in gpu2[0].split(",")]
            gpu_str = f"mem={p[1]}/49152MB util={p[2]}% temp={p[3]}°C"
        else:
            gpu_str = "GPU2 not found"

        n_ckpt = run_remote(c, f"ls -lt {REMOTE}/step_*.pt 2>/dev/null | wc -l")
        ckpt_str = f"ckpt={n_ckpt}"

        n_proc = run_remote(c, "ps -u jiaqigu | grep -c python 2>/dev/null || echo 0")
        proc_count = int(n_proc.strip()) if n_proc.strip().isdigit() else 0
        if proc_count == 0:
            status = "[WARN] process dead"
            proc_str = f"proc={proc_count}"
        else:
            proc_str = f"proc={proc_count}"

    except Exception as e:
        status = "[WARN] ssh failed"
        log_str = str(e)[:80]
    finally:
        if c:
            try:
                c.close()
            except Exception:
                pass

    ts = datetime.now().strftime("%m-%d %H:%M")
    parts = [ts, "farm04 GPU2", log_str, gpu_str, ckpt_str, proc_str]
    if status != "OK":
        parts.append(status)
    print(" | ".join(parts))


if __name__ == "__main__":
    import sys
    once = "--once" in sys.argv

    if not once:
        print(f"=== MR-RATE Monitor 启动 (每{INTERVAL//60}分钟, 卡住>{STALL_MINUTES}分钟告警) ===")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

    try:
        while True:
            check()
            if once:
                break
            print(f"[下次检查: {(datetime.now() + timedelta(seconds=INTERVAL)).strftime('%H:%M:%S')}]")
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("\n监控已停止。")
