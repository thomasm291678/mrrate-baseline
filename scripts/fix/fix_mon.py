import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=30)

# Upload fixed script
sftp = c.open_sftp()
sftp.put(
    r"C:\Users\HP\Documents\5555\check_monitor.sh",
    "/home/jiaqigu/mrrate_hidnet/check_monitor.sh")
sftp.close()

# Test run
c.exec_command("bash /home/jiaqigu/mrrate_hidnet/check_monitor.sh")

stdin, stdout, stderr = c.exec_command(
    "cat /home/jiaqigu/mrrate_hidnet/outputs/report_gen/monitor.log")
print("Monitor log:")
print(stdout.read().decode().strip())

# Check crontab active
stdin, stdout, stderr = c.exec_command("crontab -l")
print("\nCrontab:", stdout.read().decode().strip())

# training
stdin, stdout, stderr = c.exec_command(
    "tail -2 $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_*.log|head -1)")
print("\nTraining:", stdout.read().decode().strip())

c.close()
