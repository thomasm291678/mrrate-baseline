import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=30)

print("Waiting 120s for more steps...")
time.sleep(120)

stdin, stdout, stderr = c.exec_command(
    "tail -8 $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_*.log | head -1)")
lines = stdout.read().decode(errors="replace")
print(lines)

stdin, stdout, stderr = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^6'")
print("GPU6:", stdout.read().decode().strip())

# Check save files
stdin, stdout, stderr = c.exec_command(
    "ls -lh /home/jiaqigu/mrrate_hidnet/outputs/report_gen/*.pt 2>/dev/null || echo 'no pt files yet'")
print("\nSaved:", stdout.read().decode().strip())

c.close()
