import paramiko, time

time.sleep(120)

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

log = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_t2_20260713_173707.log"

s, o, e = c.exec_command(f"tail -10 {log}", timeout=10)
print(o.read().decode())

s, o, e = c.exec_command(f"grep -c ' S0' {log}", timeout=10)
print(f"Steps: {o.read().decode().strip()}")

s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep -E '^(3|7),'", timeout=10)
print(o.read().decode())

s, o, e = c.exec_command("pgrep -f 'train_v4' | wc -l", timeout=10)
print(f"Processes: {o.read().decode().strip()}")

c.close()
