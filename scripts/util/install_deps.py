import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=20)
stdin, stdout, stderr = c.exec_command(
    "/home/jiaqigu/hidnet_env/bin/pip install evaluate bert_score scikit-learn rouge_score -q",
    timeout=300)
out = stdout.read().decode(errors="replace")
err = stderr.read().decode(errors="replace")
print(out or err or "installed (no output=OK)")
c.close()
