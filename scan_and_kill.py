import paramiko

FARMS = {
    "farm02": "10.176.60.70",
    "farm04": "10.176.60.71",
    "farm05": "10.176.60.72",
}

print("=== Kill all training on all farms ===")
for name, host in FARMS.items():
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(host, username="jiaqigu", password="lijia7272", timeout=10)
        _, o, _ = c.exec_command("pkill -9 -u jiaqigu -f train_v5 2>/dev/null; pkill -9 -u jiaqigu -f train_v4 2>/dev/null; pkill -9 -u jiaqigu -f train_report 2>/dev/null; echo OK")
        print(f"  {name} ({host}): killed")
        c.close()
    except Exception as e:
        print(f"  {name} ({host}): SKIP - {e}")

print("\n=== GPU scan ===")
for name, host in FARMS.items():
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(host, username="jiaqigu", password="lijia7272", timeout=10)
        _, o, _ = c.exec_command("nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader")
        gpus = o.read().decode().strip().split("\n")
        for line in gpus:
            parts = [x.strip() for x in line.split(",")]
            idx, gpu_name, mem_used, mem_total, util = parts
            mem_used_mb = int(mem_used.split()[0])
            mem_total_mb = int(mem_total.split()[0])
            is_free = mem_used_mb < 1000
            marker = " ★ FREE" if is_free else ""
            print(f"  {name} GPU{idx} {gpu_name} {mem_used}/{mem_total} {util}{marker}")
        c.close()
    except Exception as e:
        print(f"  {name}: SKIP - {e}")
