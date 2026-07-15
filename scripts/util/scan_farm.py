import paramiko

servers = {
    "farm01": "10.176.60.68",
    "farm02": "10.176.60.69",
    "farm03": "10.176.60.70",
    "farm04": "10.176.60.71",
}

print("Scanning farm01~04 for free GPUs (<8GB used)...\n")
for name, ip in servers.items():
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(ip, username="jiaqigu", password="lijia7272", timeout=8, banner_timeout=5)
        s, o, e = c.exec_command(
            "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null",
            timeout=10)
        gpus = o.read().decode().strip()
        free = []
        for line in gpus.split("\n"):
            parts = [x.strip().replace(" MiB","") for x in line.split(",")]
            idx, used, total = int(parts[0]), int(parts[1]), int(parts[2])
            if used < 8000:
                free.append((idx, total - used))
                print(f"  {name} GPU{idx}: {used}/{total}MB  FREE={total-used}MB  ✅")
            else:
                print(f"  {name} GPU{idx}: {used}/{total}MB  BUSY")
        if not free:
            print(f"  {name}: NO FREE GPUS")
        print()
        c.close()
    except Exception as ex:
        print(f"  {name}: DOWN ({ex})\n")
