import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# All GPUs
print("=== farm04 GPUs ===")
s, o, e = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader",
    timeout=10)
print(o.read().decode())

# T1 status
print("=== T1 (GPU3) ===")
s, o, e = c.exec_command(
    "grep ' S0' /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_t1_20260713_165438.log | tail -5",
    timeout=10)
print(o.read().decode().strip())

s, o, e = c.exec_command(
    "grep -c ' S0' /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_t1_20260713_165438.log",
    timeout=10)
t1_steps = int(o.read().decode().strip())

# T2 status
print("\n=== T2 (GPU7) ===")
s, o, e = c.exec_command(
    "grep ' S0' /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_t2_20260713_173707.log | tail -5",
    timeout=10)
print(o.read().decode().strip())

s, o, e = c.exec_command(
    "grep -c ' S0' /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_t2_20260713_173707.log",
    timeout=10)
t2_steps = int(o.read().decode().strip())

total = 11123 * 3
print(f"\nT1: {t1_steps}/{total} steps ({t1_steps/total*100:.2f}%)")
print(f"T2: {t2_steps}/{total} steps ({t2_steps/total*100:.2f}%)")

# Processes
s, o, e = c.exec_command("pgrep -f 'train_v4' | wc -l", timeout=10)
print(f"train_v4 processes: {o.read().decode().strip()}")

c.close()
