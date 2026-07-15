import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

log = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v5_b27_e5_20260714_000554.log"

s, o, e = c.exec_command(f"tail -5 {log}", timeout=10)
print("=== Latest log ===")
print(o.read().decode())

s, o, e = c.exec_command(f"grep -c ' S0' {log}", timeout=10)
n = int(o.read().decode().strip())
print(f"Steps logged: {n}")

s, o, e = c.exec_command("pgrep -f 'train_v5' | wc -l", timeout=10)
print(f"Procs: {o.read().decode().strip()}")

s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep -E '^(3|0|7),'", timeout=10)
print("GPUs:\n" + o.read().decode())

s, o, e = c.exec_command("ls -lh /home/jiaqigu/mrrate_hidnet/outputs/report_gen/ | grep -E 'v5|latest' | head -8", timeout=10)
print("Checkpoints:")
print(o.read().decode())

# Also check if any other process is running
s, o, e = c.exec_command("ps -u jiaqigu -o pid,stat,comm | grep -v 'ps\\|bash\\|grep' | head -15", timeout=10)
print("All procs:")
print(o.read().decode())

c.close()
