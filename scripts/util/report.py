import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

log = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v5_encoder_20260713_204642.log"

s, o, e = c.exec_command(f"tail -8 {log}", timeout=10)
print("=== Last 8 steps ===")
print(o.read().decode())

s, o, e = c.exec_command(f"grep -c ' S0' {log}", timeout=10)
n = int(o.read().decode().strip())
print(f"Steps logged since resume: {n}")

s, o, e = c.exec_command(f"tail -1 {log}", timeout=10)
last = o.read().decode().strip()

s, o, e = c.exec_command("pgrep -f 'train_v5' | wc -l", timeout=10)
print(f"Processes: {o.read().decode().strip()}")

s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^3,'", timeout=10)
print("GPU3:", o.read().decode().strip())

# Check if training finished
s, o, e = c.exec_command(f"grep -i 'finished\|Epoch 2 done' {log} | tail -3", timeout=10)
done = o.read().decode().strip()
if done:
    print(f"\nDone: {done}")

# Check latest checkpoint
s, o, e = c.exec_command("ls -lt /home/jiaqigu/mrrate_hidnet/outputs/report_gen/v5_encoder_* 2>/dev/null | head -3", timeout=10)
print("\nCheckpoints:")
print(o.read().decode().strip())

c.close()
