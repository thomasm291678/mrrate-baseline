import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# First check current state
stdin, stdout, stderr = c.exec_command(
    "/home/jiaqigu/hidnet_env/bin/pip install evaluate scikit-learn rouge_score 2>&1 | tail -3", timeout=60)
print("Pip:", stdout.read().decode().strip())

# Check if torch is broken
stdin, stdout, stderr = c.exec_command(
    "/home/jiaqigu/hidnet_env/bin/python -c 'import torch;print(torch.cuda.is_available())' 2>&1", timeout=10)
print("Torch:", stdout.read().decode().strip(), stderr.read().decode().strip()[:200])

# If torch broken, reinstall in background
if "Error" in stdout.read().decode() + stderr.read().decode():
    print("Torch broken - reinstalling in background...")
    c.exec_command(
        "nohup /home/jiaqigu/hidnet_env/bin/pip install torch==2.5.1 --force-reinstall > /tmp/torch_reinstall.log 2>&1 &", timeout=5)

c.close()
