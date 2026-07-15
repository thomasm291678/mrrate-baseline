import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

log = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v5_b27_20260713_221953.log"

s, o, e = c.exec_command(f"tail -8 {log}", timeout=10)
print(o.read().decode())

s, o, e = c.exec_command(f"grep -c ' S0' {log}", timeout=10)
n = int(o.read().decode().strip())
print(f"Steps logged: {n}, total: 572, {n*5/572*100:.1f}%")

s, o, e = c.exec_command("pgrep -f 'train_v5' | wc -l", timeout=10)
print(f"Procs: {o.read().decode().strip()}")

s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^3,'", timeout=10)
print("GPU3:", o.read().decode().strip())

s, o, e = c.exec_command("grep 'Epoch' " + log + " | grep done", timeout=10)
done = o.read().decode().strip()
if done:
    print("DONE:", done)

c.close()
