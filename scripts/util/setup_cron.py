import paramiko, os

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=30)

# 1. Kill watchdog daemon
print("1. Killing daemon...")
c.exec_command("pkill -f watchdog_daemon 2>/dev/null; true")

# 2. Upload monitor script
print("2. Uploading monitor script...")
sftp = c.open_sftp()
sftp.put(
    r"C:\Users\HP\Documents\5555\check_monitor.sh",
    "/home/jiaqigu/mrrate_hidnet/check_monitor.sh")
sftp.close()
c.exec_command("chmod +x /home/jiaqigu/mrrate_hidnet/check_monitor.sh")

# 3. Setup crontab (every 5 min)
print("3. Setting crontab...")
cron_entry = "*/5 * * * * /home/jiaqigu/mrrate_hidnet/check_monitor.sh"

# Add to crontab (remove old entry, add new)
script = (
    'crontab -l 2>/dev/null | grep -v check_monitor | '
    '(cat; echo "' + cron_entry + '") | crontab -'
)
c.exec_command(script)

# 4. Verify
stdin, stdout, stderr = c.exec_command("crontab -l")
print("\nCrontab:\n" + stdout.read().decode().strip())

# 5. Test run
print("\nTesting monitor script...")
c.exec_command("bash /home/jiaqigu/mrrate_hidnet/check_monitor.sh")
stdin, stdout, stderr = c.exec_command(
    "cat /home/jiaqigu/mrrate_hidnet/outputs/report_gen/monitor.log 2>/dev/null | tail -6")
print("Monitor log:\n" + stdout.read().decode().strip())

# 6. Confirm training alive
stdin, stdout, stderr = c.exec_command(
    "tail -3 $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_*.log | head -1)")
print("\nTraining:\n" + stdout.read().decode().strip())

c.close()
print("\nDone. Crontab on farm01: every 5 min -> monitor.log")
