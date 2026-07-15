import paramiko, sys

servers = {
    "farm01": ("10.154.32.185", "jiaqigu", "lijia7272"),
    "farm02": ("10.154.32.115", "jiaqigu", "lijia7272"),
    "farm03": ("10.176.60.70", "jiaqigu", "lijia7272"),
    "farm04": ("10.176.60.71", "jiaqigu", "lijia7272"),
    "farm05": ("10.176.60.72", "jiaqigu", "lijia7272"),
}

for name, (ip, user, pw) in servers.items():
    print(f"\n=== {name} ({ip}) ===")
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(ip, username=user, password=pw, timeout=10)
        stdin, out, err = c.exec_command(
            "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,name --format=csv,noheader 2>/dev/null || echo NO_GPU"
        )
        gpu_info = out.read().decode().strip()
        stdin2, out2, err2 = c.exec_command(
            "ls /home/jiaqigu/mrrate_hidnet/outputs/report_gen/latest_step.pt 2>/dev/null && echo EXISTS || echo NO_CKPT"
        )
        ckpt = out2.read().decode().strip()
        stdin3, out3, err3 = c.exec_command(
            "ls /mnt/nas1/disk07/public/mr_data/MR-RATE/splits.csv 2>/dev/null && echo HAS_DATA || echo NO_DATA"
        )
        has_data = out3.read().decode().strip()
        stdin4, out4, err4 = c.exec_command(
            "test -f /home/jiaqigu/hidnet_env/bin/python && echo HAS_PYTHON || echo NO_PYTHON"
        )
        has_python = out4.read().decode().strip()
        stdin5, out5, err5 = c.exec_command(
            "ls /home/jiaqigu/mrrate_hidnet/encoder_v5.py 2>/dev/null && echo HAS_V5 || echo NO_V5"
        )
        has_v5 = out5.read().decode().strip()

        print(f"  Python: {has_python}  V5: {has_v5}  Data: {has_data}  Ckpt: {ckpt}")
        if gpu_info == "NO_GPU":
            print("  GPU: 无GPU或无nvidia-smi")
        else:
            print("  GPU:")
            for line in gpu_info.split("\n"):
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 4:
                    idx, mem_used, mem_total, util = parts[0], parts[1], parts[2], parts[3]
                    gpu_name = parts[4] if len(parts) > 4 else ""
                    try:
                        used_mb = int(mem_used.split()[0])
                        total_mb = int(mem_total.split()[0])
                        free_mb = total_mb - used_mb
                    except:
                        free_mb = "?"
                    flag = "*** FREE ***" if isinstance(free_mb, int) and free_mb > 15000 else ""
                    print(f"    GPU{idx}: {mem_used}/{mem_total}  util={util}  free={free_mb}MiB  {flag}  {gpu_name}")
        c.close()
    except Exception as e:
        print(f"  ERROR: {e}")
