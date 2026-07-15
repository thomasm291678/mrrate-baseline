import paramiko

FARMS = {
    "farm01": ("10.176.60.69", 22),
    "farm02": ("10.176.60.70", 22),
    "farm03": ("10.176.60.71", 22),
    "farm04": ("10.176.60.72", 22),
    "farm05": ("10.176.60.73", 22),
}

PASS = "lijia7272"
USER = "jiaqigu"

results = []

for name, (host, port) in FARMS.items():
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(host, username=USER, password=PASS, timeout=8)

        # GPU info
        _, o, _ = c.exec_command("nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader 2>&1")
        gpu_raw = o.read().decode().strip()

        if not gpu_raw or "NVIDIA" not in gpu_raw:
            results.append(f"{name} ({host}): NO GPU / UNREACHABLE")
            c.close()
            continue

        gpu_lines = gpu_raw.split("\n")
        # Check if jiaqigu has any process
        _, o, _ = c.exec_command("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1")
        procs = o.read().decode().strip()

        # Disk
        _, o, _ = c.exec_command("df -h /home/jiaqigu 2>&1 | tail -1")
        disk = o.read().decode().strip()

        free_gpus = []
        for line in gpu_lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                idx, gpu_name, mem_used, mem_total, util, temp = parts
                mem_used_mb = int(mem_used.split()[0].replace("MiB", ""))
                mem_total_mb = int(mem_total.split()[0].replace("MiB", ""))
                util_pct = int(util.split()[0].replace("%", "").replace("N/A", "0"))
                free_pct = 100 - (mem_used_mb / mem_total_mb * 100)
                # < 30% util and > 60% free memory
                if util_pct < 30 and free_pct > 50:
                    free_gpus.append(f"GPU{idx}: {gpu_name} util={util} mem={mem_used}/{mem_total} ({free_pct:.0f}% free) temp={temp}")

        if free_gpus:
            results.append(f"\n✅ {name} ({host}) — {len(free_gpus)} free GPU(s):")
            for g in free_gpus:
                results.append(f"     {g}")
        else:
            # Show all
            busy = []
            for line in gpu_lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    idx, gpu_name, mem_used, mem_total, util, temp = parts
                    busy.append(f"GPU{idx}: {gpu_name} util={util} mem={mem_used}/{mem_total} temp={temp}")
            results.append(f"\n❌ {name} ({host}) — all busy:")
            for b in busy:
                results.append(f"     {b}")

        if procs:
            results.append(f"   Processes: {procs[:150]}")

        c.close()
    except Exception as e:
        results.append(f"\n⚠️ {name} ({host}): {str(e)[:80]}")

print("\n".join(results))
