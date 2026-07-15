import paramiko, time

# farm04 - kill all train.py processes
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Find all train.py PIDs
s, o, e = c.exec_command("ps -u jiaqigu | grep 'train.py' | grep -v grep | awk '{print $2}'", timeout=10)
pids = o.read().decode().strip()
print(f"farm04 PIDs: {pids if pids else 'none'}")

if pids:
    c.exec_command(f"pkill -f 'train.py' 2>/dev/null; true", timeout=5)
    print("farm04: killed all train.py")

# Also kill the data workers
c.exec_command("pkill -f 'pt_data_worker' 2>/dev/null; true", timeout=5)

time.sleep(3)
s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep -E '^0,|^2,|^6,'", timeout=10)
print("farm04 GPU0/2/6:", o.read().decode().strip())
c.close()

# farm05 - kill old train.py
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=15)
s, o, e = c.exec_command("ps -u jiaqigu | grep 'train.py' | grep -v grep | awk '{print $2}'", timeout=10)
pids = o.read().decode().strip()
print(f"farm05 PIDs: {pids if pids else 'none'}")
if pids:
    c.exec_command(f"pkill -f 'train.py' 2>/dev/null; true", timeout=5)
    print("farm05: killed all train.py")
c.close()

print("\nDone - all old training processes killed")
