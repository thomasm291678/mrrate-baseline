import paramiko, time

for name, ip in [("farm04", "10.176.60.71"), ("farm05", "10.176.60.72")]:
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(ip, username="jiaqigu", password="lijia7272", timeout=10)
        
        # Kill all python training processes
        c.exec_command("pkill -9 -f 'train.py' 2>/dev/null; true", timeout=5)
        c.exec_command("pkill -9 -f 'pt_data_worker' 2>/dev/null; true", timeout=5)
        c.exec_command("pkill -9 -f 'python.*train' 2>/dev/null; true", timeout=5)
        time.sleep(3)
        
        # Fuser-kill all GPUs
        for g in range(8):
            c.exec_command(f"fuser -k /dev/nvidia{g} 2>/dev/null; true", timeout=5)
        time.sleep(8)
        
        s, o, e = c.exec_command(
            "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader", timeout=10)
        print(f"{name}:")
        for line in o.read().decode().strip().split("\n"):
            if line.strip():
                parts = line.split(",")
                gpu = parts[0].strip()
                mem = parts[1].strip()
                tag = "CLEAN" if int(mem.replace(" MiB","")) < 8000 else "DIRTY"
                print(f"  GPU{gpu}: {mem} {tag}")
        
        c.close()
    except Exception as ex:
        print(f"{name}: DOWN ({ex})")

print("\nDone")
