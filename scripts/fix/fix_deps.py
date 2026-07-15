import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Check if pip is still running
stdin, stdout, stderr = c.exec_command("ps aux | grep pip | grep -v grep", timeout=10)
print("PIP:", stdout.read().decode().strip() or "not running")

# Try import
stdin, stdout, stderr = c.exec_command(
    "/home/jiaqigu/hidnet_env/bin/python -c 'from evaluate import load; print(\"OK\")' 2>&1", timeout=10)
print("Import:", stdout.read().decode().strip()[:200])

# If failed, install now and wait
if "OK" not in stdout.read().decode():
    print("Installing now...")
    c.exec_command(
        "nohup /home/jiaqigu/hidnet_env/bin/pip install evaluate bert_score scikit-learn rouge_score -q > /tmp/pip_install.log 2>&1 &", timeout=5)
    time.sleep(60)
    stdin2, stdout2, stderr2 = c.exec_command("cat /tmp/pip_install.log", timeout=10)
    print("Install log:", stdout2.read().decode()[:500])

c.close()
