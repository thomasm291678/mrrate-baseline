import paramiko, os

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=30)

stdin, stdout, stderr = c.exec_command(
    "tail -30 $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_*.log | head -1)")
print("=== Training Log ===")
print(stdout.read().decode(errors="replace"))

stdin, stdout, stderr = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,memory.total,temperature.gpu --format=csv,noheader | grep '^6'")
print("GPU6:", stdout.read().decode().strip())

stdin, stdout, stderr = c.exec_command(
    "ps -u jiaqigu | grep python | head -3")
print("Python:", stdout.read().decode().strip())

c.close()

# Check local download
local_model = r"C:\Users\HP\Documents\5555\weights\best_model.pt"
if os.path.exists(local_model):
    sz = os.path.getsize(local_model)
    print(f"\nLocal weight download: {sz/1024**3:.2f}GB / 9.1GB")
else:
    print("\nLocal weight download: file not found")
