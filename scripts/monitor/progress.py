import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

log = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v5_all_encoder_20260713_193843.log"

s, o, e = c.exec_command(f"tail -5 {log}", timeout=10)
print(o.read().decode())

s, o, e = c.exec_command(f"grep -c ' S0' {log}", timeout=10)
n_steps = int(o.read().decode().strip())
eff = n_steps * 10
pct = eff / 22246 * 100

s, o, e = c.exec_command(f"grep ' S0' {log} | head -2 | grep -oP '\\d{2}:\\d{2}:\\d{2}'", timeout=10)
first_ts = o.read().decode().strip().split('\n')[0]
s, o, e = c.exec_command(f"grep ' S0' {log} | tail -1 | grep -oP '\\d{2}:\\d{2}:\\d{2}'", timeout=10)
last_ts = o.read().decode().strip()

def s_(t): h,m,s=t.split(':'); return int(h)*3600+int(m)*60+int(s)
elapsed = s_(last_ts) - s_(first_ts)
speed = elapsed / max(n_steps, 1)
remaining = (22246 - n_steps) * speed / 3600

print(f"\nStep: {n_steps}/22246 ({pct:.1f}%) | eff_steps: {eff}")
print(f"Speed: {speed:.1f}s/step | Elapsed: {elapsed/3600:.1f}h | Remaining: ~{remaining:.1f}h")
print(f"ETA completion: ~{remaining:.1f}h from now")

s, o, e = c.exec_command("pgrep -f 'train_v5' | wc -l", timeout=10)
print(f"Processes: {o.read().decode().strip()}")

s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^3,'", timeout=10)
print("GPU3:", o.read().decode().strip())

c.close()
