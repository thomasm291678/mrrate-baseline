import paramiko, time

for name, ip in [("farm04", "10.176.60.71"), ("farm05", "10.176.60.72")]:
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(ip, username="jiaqigu", password="lijia7272", timeout=10)
        
        # Use setsid to detach from SSH so pkill doesn't kill our session
        c.exec_command(
            "setsid bash -c 'sleep 1 && pkill -9 -u jiaqigu 2>/dev/null; true' &",
            timeout=5)
        print(f"{name}: pkill fired (setsid)")
        c.close()
    except Exception as ex:
        print(f"{name}: {ex}")

print("Waiting 20s for kills to take effect...")
time.sleep(20)

# Reconnect and check
for name, ip in [("farm04", "10.176.60.71"), ("farm05", "10.176.60.72")]:
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(ip, username="jiaqigu", password="lijia7272", timeout=10)
        s, o, e = c.exec_command(
            "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader", timeout=10)
        print(f"\n{name}:")
        for line in o.read().decode().strip().split("\n"):
            if line.strip():
                parts = line.split(",")
                gpu = parts[0].strip()
                mem = parts[1].strip().replace(" MiB", "")
                tag = "CLEAN" if int(mem) < 5000 else "DIRTY"
                print(f"  GPU{gpu}: {mem} MiB {tag}")
        c.close()
    except Exception as ex:
        print(f"{name}: reconnecting... {ex}")
        # Retry once
        time.sleep(5)
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(ip, username="jiaqigu", password="lijia7272", timeout=15)
            s, o, e = c.exec_command(
                "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader", timeout=10)
            print(f"{name} (retry):")
            for line in o.read().decode().strip().split("\n"):
                if line.strip():
                    print(f"  GPU{line.split(',')[0].strip()}: {line.split(',')[1].strip()}")
            c.close()
        except:
            print(f"{name}: still down")

print("\nDone")
