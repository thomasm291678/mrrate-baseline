import paramiko, time

for name, ip in [("farm04", "10.176.60.71"), ("farm05", "10.176.60.72")]:
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(ip, username="jiaqigu", password="lijia7272", timeout=10)
        
        # Kill ALL of jiaqigu's user processes
        c.exec_command("pkill -9 -u jiaqigu 2>/dev/null; true", timeout=5)
        c.exec_command("killall -9 -u jiaqigu 2>/dev/null; true", timeout=5)
        time.sleep(5)
        
        s, o, e = c.exec_command(
            "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader", timeout=10)
        print(f"{name}:")
        for line in o.read().decode().strip().split("\n"):
            if line.strip():
                parts = line.split(",")
                gpu = parts[0].strip()
                mem = parts[1].strip()
                tag = "CLEAN" if int(mem.replace(" MiB","")) < 5000 else "DIRTY"
                print(f"  GPU{gpu}: {mem} {tag}")
        
        s, o, e = c.exec_command("ps -u jiaqigu | wc -l", timeout=10)
        print(f"  procs: {o.read().decode().strip()}")
        c.close()
    except Exception as ex:
        print(f"{name}: {ex}")

print("\nDone")
