import paramiko, os, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=30)

# 1. Kill old watchdogs
print("1. Killing old watchdogs...")
o, e, rc = None, None, 0
stdin, stdout, stderr = c.exec_command("pkill -f watchdog_daemon 2>/dev/null; pkill -f 'watchdog.sh' 2>/dev/null; true")
time.sleep(2)

# 2. Upload daemon
print("2. Uploading watchdog daemon...")
sftp = c.open_sftp()
sftp.put(
    r"C:\Users\HP\Documents\5555\watchdog_daemon.py",
    "/home/jiaqigu/mrrate_hidnet/watchdog_daemon.py")
sftp.close()
c.exec_command("chmod +x /home/jiaqigu/mrrate_hidnet/watchdog_daemon.py")

# 3. Start daemon on server
print("3. Starting daemon...")
c.exec_command(
    "cd /home/jiaqigu/mrrate_hidnet && "
    "nohup python3 watchdog_daemon.py >> outputs/report_gen/watchdog_daemon.log 2>&1 &")
time.sleep(5)

# 4. Verify
stdin, stdout, stderr = c.exec_command(
    "ps -u jiaqigu | grep -E 'watchdog_daemon|train.py' | grep -v grep")
print("4. Running:")
print(stdout.read().decode().strip())

# Check daemon log
stdin, stdout, stderr = c.exec_command(
    "cat /home/jiaqigu/mrrate_hidnet/outputs/report_gen/watchdog.log 2>/dev/null | tail -10")
print("\nDaemon log:")
print(stdout.read().decode().strip())

# Check latest training
stdin, stdout, stderr = c.exec_command(
    "tail -5 $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_*.log | head -1)")
print("\nTraining:")
print(stdout.read().decode().strip())

c.close()
print("\nDone! Daemon running on farm01, checking every 5min.")
