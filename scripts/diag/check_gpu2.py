import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

log = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v5_encoder_20260713_220133.log"

s, o, e = c.exec_command(f"tail -5 {log}", timeout=10)
print(o.read().decode())

s, o, e = c.exec_command(f"grep -c ' S0' {log}", timeout=10)
n = int(o.read().decode().strip())
print(f"Steps: {n}")

for i in range(2):
    s, o, e = c.exec_command("nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | grep '^3,'", timeout=10)
    print(f"GPU3 [{i}]: {o.read().decode().strip()}")

s, o, e = c.exec_command("pgrep -f 'train_v5' | wc -l", timeout=10)
print(f"Procs: {o.read().decode().strip()}")

c.close()
