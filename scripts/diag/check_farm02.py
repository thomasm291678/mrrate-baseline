import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.154.32.115", username="jiaqigu", password="lijia7272", timeout=15)

# Check Python env
stdin, stdout, stderr = c.exec_command("which python3 && python3 --version", timeout=10)
print("Python:", stdout.read().decode().strip())

# Check torch
stdin, stdout, stderr = c.exec_command(
    "python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())' 2>&1", timeout=10)
print("Torch:", stdout.read().decode().strip()[:200] + " " + stderr.read().decode().strip()[:200])

# Check if evaluate available
stdin, stdout, stderr = c.exec_command(
    "pip3 install evaluate scikit-learn rouge_score -q 2>&1 | tail -3", timeout=60)
print("Install:", stdout.read().decode().strip())

# Check project exists
stdin, stdout, stderr = c.exec_command("ls /home/jiaqigu/mrrate_hidnet/scripts/train.py 2>/dev/null && echo EXISTS", timeout=10)
print("Project:", stdout.read().decode().strip())

c.close()
