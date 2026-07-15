import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Grab last 5 step lines to measure real speed
s, o, e = c.exec_command(
    "grep ' S00' /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_t1_20260713_163707.log | tail -8",
    timeout=10)
print("=== Recent steps ===")
print(o.read().decode().strip())

s, o, e = c.exec_command(
    "grep ' S00' /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_t1_20260713_163707.log | wc -l",
    timeout=10)
n = int(o.read().decode().strip())
total = 11123 * 3
pct = n / total * 100
print(f"\nEffective steps logged: {n} / {total} ({pct:.2f}%)")

# GPU
s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep '^3,'", timeout=10)
print("GPU3:", o.read().decode().strip())

# tmux alive?
s, o, e = c.exec_command("tmux ls 2>/dev/null", timeout=10)
print("tmux:", o.read().decode().strip())

c.close()
