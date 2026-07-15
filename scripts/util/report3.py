import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=30)

log = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v5_encoder_20260713_204642.log"

s, o, e = c.exec_command(f"tail -8 {log}", timeout=10)
print(o.read().decode())

s, o, e = c.exec_command(f"grep -c ' S0' {log}", timeout=10)
n = int(o.read().decode().strip())
print(f"Steps: {n}")

s, o, e = c.exec_command("pgrep -f 'train_v5' | wc -l", timeout=10)
print(f"Processes: {o.read().decode().strip()}")

s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^3,'", timeout=10)
print("GPU3:", o.read().decode().strip())

s, o, e = c.exec_command("grep 'Epoch 2 done' " + log, timeout=10)
done = o.read().decode().strip()
if done:
    print("DONE:", done)

s, o, e = c.exec_command("ls -lt /home/jiaqigu/mrrate_hidnet/outputs/report_gen/v5_encoder_* 2>/dev/null | head -3", timeout=10)
print("CKPT:", o.read().decode().strip())

c.close()
