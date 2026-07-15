import paramiko

host = "10.176.60.70"
user = "jiaqigu"
password = "lijia7272"

print(f"[*] Connecting to {host} ...")
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    client.connect(hostname=host, username=user, password=password, timeout=15)
    print("[+] Connected.\n")
except Exception as e:
    print(f"[-] Connection failed: {e}")
    exit(1)

print("=" * 60)
print("  TMUX Training Progress (per GPU)")
print("=" * 60)

for gpu_id in range(4):
    session_name = f"gpu{gpu_id}"
    cmd = f"tmux capture-pane -t {session_name} -p 2>/dev/null | grep -E 'loss=|E0[0-9]' | tail -1"
    stdin, stdout, stderr = client.exec_command(cmd, timeout=10)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    print(f"\n[GPU {gpu_id}] tmux={session_name}")
    if err:
        print(f"  stderr: {err}")
    if out:
        print(f"  >> {out}")
    else:
        print("  (no grep match)")

print("\n" + "=" * 60)
print("  NVIDIA-SMI")
print("=" * 60)
stdin, stdout, stderr = client.exec_command(
    "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | head -4",
    timeout=10
)
smi_out = stdout.read().decode("utf-8", errors="replace").strip()
for line in smi_out.split("\n"):
    print(f"  {line}")

print("\n" + "=" * 60)
print("  train_v5 process count per GPU")
print("=" * 60)
stdin, stdout, stderr = client.exec_command(
    "ps aux | grep 'train_v5.py' | grep -v grep | grep -oP '--gpu \d+' | sort | uniq -c",
    timeout=10
)
proc_out = stdout.read().decode("utf-8", errors="replace").strip()
if proc_out:
    for line in proc_out.split("\n"):
        print(f"  {line}")
else:
    print("  No train_v5 processes — COMPLETE.")

# Check completion time from all tmux sessions
print("\n" + "=" * 60)
print("  Last 3 lines of each tmux session")
print("=" * 60)
for gpu_id in range(4):
    session_name = f"gpu{gpu_id}"
    cmd = f"tmux capture-pane -t {session_name} -p 2>/dev/null | tail -3"
    stdin, stdout, stderr = client.exec_command(cmd, timeout=10)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    print(f"\n[GPU {gpu_id}] last lines:")
    for line in out.split("\n"):
        print(f"  {line}")

client.close()
print("\n[*] Done.")
