import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Check OOM killer
s, o, e = c.exec_command("dmesg | grep -i 'killed\|oom\|out of memory' | tail -10 2>/dev/null; echo '---'; dmesg | grep -i 'train.py\|python' | tail -5 2>/dev/null", timeout=10)
print(o.read().decode())

# Check disk
s, o, e = c.exec_command("df -h /home/jiaqigu/ | tail -1; echo '---'; df -h /mnt/nas1/disk07/public/qi/ | tail -1", timeout=10)
print(o.read().decode())

# Check how many step_*.pt on disk
s, o, e = c.exec_command("ls -lh /home/jiaqigu/mrrate_hidnet/outputs/report_gen/step_*.pt 2>/dev/null | wc -l; echo 'files'; ls -lh /home/jiaqigu/mrrate_hidnet/outputs/report_gen/step_*.pt 2>/dev/null", timeout=10)
print(o.read().decode())

c.close()
