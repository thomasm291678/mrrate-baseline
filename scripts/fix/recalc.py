import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

t1 = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_t1_20260713_165438.log"
t2 = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_t2_20260713_173707.log"
total = 33369

# T1: measure 10-step window
s, o, e = c.exec_command(
    f"grep ' S0' {t1} | tail -11 | head -10 | grep -oP '\\[\\K\\d{{2}}:\\d{{2}}:\\d{{2}}'",
    timeout=10)
t1_times = o.read().decode().strip().split('\n')
s, o, e = c.exec_command(
    f"grep ' S0' {t1} | tail -1 | grep -oP '\\[\\K\\d{{2}}:\\d{{2}}:\\d{{2}}'",
    timeout=10)
t1_last = o.read().decode().strip()
s, o, e = c.exec_command(f"grep -c ' S0' {t1}", timeout=10)
t1_n = int(o.read().decode().strip()) * 10

def sec(t):
    h,m,s = t.split(':')
    return int(h)*3600+int(m)*60+int(s)

if len(t1_times) >= 2:
    t1_elapsed = sec(t1_last) - sec(t1_times[0])
    t1_n_window = (len(t1_times) - 1) * 10
    t1_spd = t1_elapsed / t1_n_window if t1_n_window > 0 else 0
    t1_rem = (total - t1_n) * t1_spd / 3600
    print(f"T1: {t1_n}/{total} ({t1_n/total*100:.1f}%) | {t1_spd:.1f}s/opt_step | ETA: {t1_rem:.1f}h")
else:
    print("T1: not enough data")

# T2 
s, o, e = c.exec_command(
    f"grep ' S0' {t2} | grep -oP '\\[\\K\\d{{2}}:\\d{{2}}:\\d{{2}}'",
    timeout=10)
t2_times = o.read().decode().strip().split('\n')
s, o, e = c.exec_command(
    f"grep ' S0' {t2} | tail -1 | grep -oP '\\[\\K\\d{{2}}:\\d{{2}}:\\d{{2}}'",
    timeout=10)
t2_last = o.read().decode().strip()
s, o, e = c.exec_command(f"grep -c ' S0' {t2}", timeout=10)
t2_n = int(o.read().decode().strip()) * 10

if len(t2_times) >= 2:
    t2_elapsed = sec(t2_last) - sec(t2_times[0])
    t2_n_window = (len(t2_times) - 1) * 10
    t2_spd = t2_elapsed / t2_n_window if t2_n_window > 0 else 0
    t2_rem = (total - t2_n) * t2_spd / 3600
    print(f"T2: {t2_n}/{total} ({t2_n/total*100:.1f}%) | {t2_spd:.1f}s/opt_step | ETA: {t2_rem:.1f}h")
else:
    print("T2: not enough data")

# Script's own ETA
s, o, e = c.exec_command(f"grep ' S0' {t1} | tail -1", timeout=10)
print(f"\nT1 script ETA from last log: {o.read().decode().strip()[-20:]}")
s, o, e = c.exec_command(f"grep ' S0' {t2} | tail -1", timeout=10)
print(f"T2 script ETA from last log: {o.read().decode().strip()[-20:]}")

c.close()
