import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Continuous GPU check — 3 samples
for i in range(3):
    s, o, e = c.exec_command(
        "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^3,'",
        timeout=10)
    print(f"[{i}] GPU3: {o.read().decode().strip()}")
    
    s, o, e = c.exec_command(
        "tail -2 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v5_encoder_20260713_204642.log",
        timeout=10)
    print(f"    {o.read().decode().strip().split(chr(10))[-1]}")

s, o, e = c.exec_command("pgrep -f 'train_v5' | wc -l", timeout=10)
print(f"Processes: {o.read().decode().strip()}")

c.close()
