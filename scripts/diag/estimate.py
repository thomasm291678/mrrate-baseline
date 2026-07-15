import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# T1 log
t1_log = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_t1_20260713_165438.log"
t2_log = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_t2_20260713_173707.log"

total = 11123 * 3

# T1: measure real speed from timestamps
s, o, e = c.exec_command(f"grep ' S0' {t1_log} | head -4", timeout=10)
print("=== T1 first 4 steps ===")
print(o.read().decode().strip())

s, o, e = c.exec_command(f"grep ' S0' {t1_log} | tail -4", timeout=10)
print("\n=== T1 last 4 steps ===")
print(o.read().decode().strip())

s, o, e = c.exec_command(f"grep -c ' S0' {t1_log}", timeout=10)
t1_steps = int(o.read().decode().strip())

s, o, e = c.exec_command(f"head -1 {t1_log} | grep -oP '\\d{2}:\\d{2}:\\d{2}'", timeout=10)
t1_start = o.read().decode().strip()
s, o, e = c.exec_command(f"grep ' S0' {t1_log} | tail -1 | grep -oP '\\d{2}:\\d{2}:\\d{2}'", timeout=10)
t1_last = o.read().decode().strip()
print(f"\nT1 start: {t1_start}, last step: {t1_last}")

# T2
s, o, e = c.exec_command(f"grep ' S0' {t2_log} | tail -4", timeout=10)
print("\n=== T2 last 4 steps ===")
print(o.read().decode().strip())

s, o, e = c.exec_command(f"grep -c ' S0' {t2_log}", timeout=10)
t2_steps = int(o.read().decode().strip())

s, o, e = c.exec_command(f"head -1 {t2_log} | grep -oP '\\d{2}:\\d{2}:\\d{2}'", timeout=10)
t2_start = o.read().decode().strip()
s, o, e = c.exec_command(f"grep ' S0' {t2_log} | tail -1 | grep -oP '\\d{2}:\\d{2}:\\d{2}'", timeout=10)
t2_last = o.read().decode().strip()
print(f"\nT2 start: {t2_start}, last step: {t2_last}")

# Calculate T1 speed from last 30 steps
s, o, e = c.exec_command(
    f"grep ' S0' {t1_log} | tail -30 | "
    "awk -F'[][]' '{print $2}' | head -1",
    timeout=10)
t1_30_start = o.read().decode().strip()
s, o, e = c.exec_command(
    f"grep ' S0' {t1_log} | tail -30 | "
    "awk -F'[][]' '{print $2}' | tail -1",
    timeout=10)
t1_30_end = o.read().decode().strip()

print(f"\nT1 30-step window: {t1_30_start} → {t1_30_end}")

# Calculate T2 speed from all steps
s, o, e = c.exec_command(
    f"grep ' S0' {t2_log} | "
    "awk -F'[][]' '{print $2}' | head -1",
    timeout=10)
t2_first = o.read().decode().strip()
s, o, e = c.exec_command(
    f"grep ' S0' {t2_log} | "
    "awk -F'[][]' '{print $2}' | tail -1",
    timeout=10)
t2_last_ts = o.read().decode().strip()

print(f"T2 window: {t2_first} → {t2_last_ts}")

def to_seconds(ts):
    h,m,s = ts.split(':')
    return int(h)*3600 + int(m)*60 + int(s)

t1_dur = to_seconds(t1_30_end) - to_seconds(t1_30_start)
t1_speed = t1_dur / 30 if t1_dur > 0 else 999
t1_remaining = (total - t1_steps * 10) * t1_speed / 10  # 10 effective steps per log
t1_eta_h = t1_remaining / 3600

t2_dur = to_seconds(t2_last_ts) - to_seconds(t2_first)
t2_n = t2_steps * 10 - 10  # steps between first and last logged
t2_speed = t2_dur / max(t2_n, 1)
t2_remaining = (total - t2_steps * 10) * t2_speed / 10
t2_eta_h = t2_remaining / 3600

print(f"\n{'='*50}")
print(f"T1: {t1_steps*10}/{total} steps ({t1_steps*10/total*100:.1f}%) | speed: {t1_speed:.1f}s/step | remaining: {t1_eta_h:.1f}h")
print(f"T2: {t2_steps*10}/{total} steps ({t2_steps*10/total*100:.1f}%) | speed: {t2_speed:.1f}s/step | remaining: {t2_eta_h:.1f}h")
print(f"{'='*50}")

c.close()
