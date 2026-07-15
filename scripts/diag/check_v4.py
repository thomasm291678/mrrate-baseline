import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Latest log
s, o, e = c.exec_command(
    "ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_gpu3_*.log 2>/dev/null | head -1",
    timeout=10)
log = o.read().decode().strip()

# Last 15 lines
s, o, e = c.exec_command(f"tail -20 {log}", timeout=10)
print(o.read().decode())

# Current step progress
s, o, e = c.exec_command(f"grep ' S0' {log} | tail -5", timeout=10)
print("\n--- Recent steps ---")
print(o.read().decode().strip())

# Epoch summaries
s, o, e = c.exec_command(f"grep -E 'Epoch [0-9]+ loss=|Eval' {log} 2>/dev/null | tail -10", timeout=10)
out = o.read().decode().strip()
if out:
    print("\n--- Epoch summaries ---")
    print(out)

# GPU
s, o, e = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^3,'",
    timeout=10)
print("\nGPU3:", o.read().decode().strip())

# Progress estimation
s, o, e = c.exec_command(f"grep ' S0' {log} | wc -l", timeout=10)
n_steps = int(o.read().decode().strip())
steps_per_epoch = 17797
epoch = (n_steps // steps_per_epoch) + 1
pct = 100 * n_steps / (steps_per_epoch * 5)
print(f"\nProgress: {n_steps} steps logged, ~epoch {epoch}/5 ({pct:.1f}%)")

c.close()
