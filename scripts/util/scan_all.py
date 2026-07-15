import paramiko

servers = {
    "farm01": ("10.154.32.185", "jiaqigu", "lijia7272"),
    "farm02": ("10.154.32.115", "jiaqigu", "lijia7272"),
    "farm03": ("10.176.60.70", "jiaqigu", "lijia7272"),
    "farm04": ("10.176.60.71", "jiaqigu", "lijia7272"),
    "farm05": ("10.176.60.72", "jiaqigu", "lijia7272"),
}

for name, (host, user, pwd) in servers.items():
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(host, username=user, password=pwd, timeout=10)
        stdin, stdout, stderr = c.exec_command(
            "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,name --format=csv,noheader",
            timeout=10)
        lines = stdout.read().decode().strip().split("\n")
        print(f"=== {name} ({host}) ===")
        free_gpus = []
        for line in lines:
            parts = [x.strip() for x in line.split(",", 4)]
            idx = parts[0]
            used = int(parts[1].split()[0])
            total = int(parts[2].split()[0])
            util = parts[3].strip()
            gpu_name = parts[4].strip() if len(parts) > 4 else ""
            free_mb = total - used
            status = "FREE" if used < 5000 and util == "0 %" else "used"
            tag = " ★" if status == "FREE" else ""
            print(f"  GPU{idx}: {used}/{total} MiB {util:>5s} [{gpu_name}]{tag}")
            if status == "FREE":
                free_gpus.append(idx)
        stdin, stdout, stderr = c.exec_command(
            "ps -u jiaqigu | grep train.py | grep -v grep | wc -l", timeout=5)
        pcount = stdout.read().decode().strip()
        print(f"  train.py procs: {pcount}  FREE GPUs: {free_gpus}")
        c.close()
    except Exception as e:
        print(f"=== {name} ({host}): OFFLINE ({e}) ===")
    print()
