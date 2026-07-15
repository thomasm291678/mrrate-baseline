import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Kill GPU1 processes
c.exec_command("fuser -k /dev/nvidia1 2>/dev/null; true", timeout=5)
time.sleep(5)

# Check if torch works
PY = "/home/jiaqigu/hidnet_env/bin/python"
stdin, stdout, stderr = c.exec_command(
    f"{PY} -c 'import torch; print(torch.__version__, torch.cuda.is_available())'", timeout=10)
print("Torch:", stdout.read().decode().strip() + " " + stderr.read().decode().strip()[:200])

# If torch broken, reinstall
if "False" in stdout.read().decode() or "Error" in stderr.read().decode():
    print("Torch broken, reinstalling...")
    stdin, stdout, stderr = c.exec_command(
        f"/home/jiaqigu/hidnet_env/bin/pip install torch==2.5.1 --force-reinstall -q 2>&1 | tail -3",
        timeout=120)
    print(stdout.read().decode())

c.close()
