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

# Ensure old crontab has this script
c.exec_command(
    'crontab -l 2>/dev/null | grep -v check_monitor | '
    '(cat; echo "*/5 * * * * /home/jiaqigu/mrrate_hidnet/check_monitor.sh") | crontab -')

# Manually trigger once
c.exec_command("/bin/bash /home/jiaqigu/mrrate_hidnet/check_monitor.sh")

stdin, stdout, stderr = c.exec_command("cat /home/jiaqigu/mrrate_hidnet/outputs/report_gen/monitor.log")
print(stdout.read().decode())

stdin, stdout, stderr = c.exec_command("crontab -l")
print("CRON:", stdout.read().decode().strip())

c.close()
