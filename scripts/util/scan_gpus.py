import paramiko

servers = {
    "farm02": "10.176.60.69",
    "farm03": "10.176.60.70", 
    "farm04": "10.176.60.71",
    "farm05": "10.176.60.72",
    "farm06": "10.176.60.73",
}

for name, ip in servers.items():
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(ip, username="jiaqigu", password="lijia7272", timeout=8, banner_timeout=5)
        s, o, e = c.exec_command(
            "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null",
            timeout=10)
        gpus = o.read().decode().strip()
        free = [l for l in gpus.split("\n") if l.strip() and int(l.split(",")[1].strip().replace(" MiB","")) < 8000]
        print(f"{name} ({ip}): {len(free)} free GPUs")
        for l in gpus.split("\n"):
            if l.strip():
                parts = [x.strip().replace(" MiB","").replace(" %","") for x in l.split(",")]
                mem_used = int(parts[1])
                mem_total = int(parts[2])
                free_mb = mem_total - mem_used
                tag = "FREE" if mem_used < 8000 else "BUSY"
                print(f"  GPU{parts[0]}: {mem_used}/{mem_total}MB ({free_mb}MB free) util={parts[3]}% {tag}")
        c.close()
    except Exception as ex:
        print(f"{name} ({ip}): DOWN ({ex})")
    print()
