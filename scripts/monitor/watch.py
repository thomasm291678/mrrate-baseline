import paramiko, time, sys, os
from datetime import datetime

HOST = "10.176.60.71"
USER = "jiaqigu"
PWD = "lijia7272"
REMOTE_DIR = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen"
CHECK_INTERVAL = 300

def ssh_connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PWD, timeout=15)
    return c

def get_training_status(c):
    cmds = [
        f"ls -t {REMOTE_DIR}/train_v3_*.log 2>/dev/null | head -1",
        "ps -u jiaqigu | grep 'train.py' | grep -v grep | wc -l",
        "nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader",
    ]
    stdin, stdout, stderr = c.exec_command("; ".join(cmds), timeout=15)
    out = stdout.read().decode(errors="replace").strip().split("\n")
    log_file = out[0].strip() if len(out) > 0 and out[0] else None
    proc_count = int(out[1].strip()) if len(out) > 1 else 0
    gpu_lines = out[2:] if len(out) > 2 else []
    return log_file, proc_count, gpu_lines

def get_latest_metrics(c, log_file):
    if not log_file:
        return None
    stdin, stdout, stderr = c.exec_command(f"tail -3 {log_file}", timeout=10)
    lines = stdout.read().decode(errors="replace").strip().split("\n")
    return [l.strip() for l in lines if l.strip()]

def get_saved_checkpoints(c):
    stdin, stdout, stderr = c.exec_command(
        f"ls -lt {REMOTE_DIR}/step_*.pt 2>/dev/null | head -5", timeout=10)
    lines = stdout.read().decode(errors="replace").strip().split("\n")
    return [l.strip() for l in lines if l.strip()]

def get_eval_results(c):
    stdin, stdout, stderr = c.exec_command(
        f"grep -E 'BLEU|Diversity|Evaluating' {REMOTE_DIR}/train_v3_*.log 2>/dev/null | tail -10",
        timeout=10)
    return stdout.read().decode(errors="replace").strip()

def get_last_save(log_file):
    c = ssh_connect()
    stdin, stdout, stderr = c.exec_command(f"grep 'Saved' {log_file} 2>/dev/null | tail -3", timeout=10)
    out = stdout.read().decode(errors="replace").strip()
    c.close()
    return out

def format_train_line(line):
    parts = line.split()
    try:
        epoch_info = parts[0].strip("[]")
        pct = parts[1] if len(parts) > 1 else ""
        step = parts[2] if len(parts) > 2 else ""
        loss = parts[3].split("=")[1] if len(parts) > 3 and "=" in parts[3] else "?"
        lr_enc = parts[4].split("=")[1].split("/")[0] if len(parts) > 4 else "?"
        mem = parts[6].split("=")[1] if len(parts) > 6 else "?"
        eta = parts[7].split("=")[1] if len(parts) > 7 else "?"
        return f"  {epoch_info} {pct} {step} loss={loss} lr={lr_enc} mem={mem} eta={eta}"
    except:
        return f"  {line}"

def main():
    os.system("cls" if os.name == "nt" else "clear")
    print("=" * 60)
    print(f"  MR-RATE Training Monitor — checking every {CHECK_INTERVAL}s")
    print(f"  farm04 GPU2  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    prev_step = None
    stuck_count = 0
    total_checks = 0

    while True:
        total_checks += 1
        try:
            c = ssh_connect()
            log_file, proc_count, gpu_lines = get_training_status(c)
            metrics = get_latest_metrics(c, log_file)
            ckpts = get_saved_checkpoints(c)
            c.close()

            now_str = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{now_str}] Check #{total_checks}")

            # GPU status
            gpu2_info = "OFFLINE"
            for line in gpu_lines:
                if line.startswith("2,"):
                    parts = [x.strip().replace(" MiB","").replace(" %","") for x in line.split(",")]
                    if len(parts) >= 4:
                        gpu2_info = f"mem={parts[1]}/49152 MiB util={parts[2]}% temp={parts[3]}C"
            print(f"  GPU2: {gpu2_info}")

            # Process count
            if proc_count == 0:
                print("[ALERT] No train.py process found!")
            else:
                print(f"  Process: {proc_count} train.py running")

            # Training metrics
            if metrics:
                for line in metrics:
                    print(format_train_line(line))
                # Detect stuck
                curr_step = None
                for line in metrics:
                    if "S0" in line:
                        try:
                            curr_step = int(line.split("S0")[1].split("]")[0])
                        except:
                            pass
                if curr_step is not None:
                    if curr_step == prev_step:
                        stuck_count += 1
                        if stuck_count >= 3:
                            print(f"[WARN] Step stuck at S{curr_step} for {stuck_count * CHECK_INTERVAL}s!")
                    else:
                        stuck_count = 0
                    prev_step = curr_step
            else:
                print("  No training output yet")

            # Checkpoints
            if ckpts:
                print(f"  Saved checkpoints ({len(ckpts)}):")
                for ck in ckpts[:5]:
                    print(f"    {ck}")
            else:
                save_info = get_last_save(log_file) if log_file else ""
                if save_info:
                    print(f"  Last save: {save_info}")
                else:
                    print("  No checkpoints yet")

            # Eval results
            eval_out = get_eval_results(ssh_connect())
            if eval_out:
                print(f"\n  Evaluation:\n{eval_out}")

        except Exception as e:
            print(f"[ERROR] {e}")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
