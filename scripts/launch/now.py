import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

log = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_t1_20260713_165438.log"

s, o, e = c.exec_command(f"tail -8 {log}", timeout=10)
print(o.read().decode())

s, o, e = c.exec_command(f"grep -c ' S0' {log}", timeout=10)
n_steps = int(o.read().decode().strip())
total = 11123 * 3
pct = n_steps / total * 100
print(f"Steps: {n_steps}/{total} ({pct:.1f}%), ETA ~{total/n_steps * (n_steps*7.7)/3600:.1f}h total, ~{(total-n_steps)*7.7/3600:.1f}h remaining")

s, o, e = c.exec_command("pgrep -f 'train_v4' | wc -l", timeout=10)
print(f"Processes: {o.read().decode().strip()}")

s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^3,'", timeout=10)
print("GPU3:", o.read().decode().strip())

c.close()
