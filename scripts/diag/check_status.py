import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)
stdin, stdout, stderr = c.exec_command(
    "tail -15 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_eval_20260713_000116.log", timeout=10)
print(stdout.read().decode())
c.close()
