import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.70", username="jiaqigu", password="lijia7272", timeout=15)

# Check if train_v4.py is synced on farm03
s, o, e = c.exec_command("ls /home/jiaqigu/mrrate_hidnet/scripts/train_v4.py 2>/dev/null && echo 'EXISTS' || echo 'MISSING'", timeout=10)
print(f"train_v4.py: {o.read().decode().strip()}")

# Check data
s, o, e = c.exec_command("ls /mnt/nas1/disk07/public/mr_data/MR-RATE/mp4/ 2>/dev/null | head -3", timeout=10)
print(f"Data: {o.read().decode().strip()}")

# Check qwen weights
s, o, e = c.exec_command("ls /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct/ 2>/dev/null | head -3", timeout=10)
print(f"Qwen: {o.read().decode().strip()}")

c.close()
