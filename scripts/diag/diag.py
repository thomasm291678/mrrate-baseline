import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Check current state
stdin, stdout, stderr = c.exec_command(
    "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader && "
    "echo '---' && "
    "ps -u jiaqigu | grep train.py | grep -v grep", timeout=10)
print(stdout.read().decode())

# Check if evaluate install corrupted torch
stdin, stdout, stderr = c.exec_command(
    "ls -lt /home/jiaqigu/hidnet_env/lib/python3.12/site-packages/ | head -5", timeout=10)
print("Recent pkgs:", stdout.read().decode())

c.close()
