import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Check save_interval config
stdin, stdout, stderr = c.exec_command(
    "grep 'save_interval' $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v3_*.log 2>/dev/null | head -1 | tail -1) | head -1", timeout=10)
print("Config:", stdout.read().decode().strip()[:200])

# Check newer checkpoint files
stdin, stdout, stderr = c.exec_command(
    "ls -lt /home/jiaqigu/mrrate_hidnet/outputs/report_gen/v3_ckpt* 2>/dev/null && echo '---' && ls -lt /home/jiaqigu/mrrate_hidnet/outputs/report_gen/checkpoint* 2>/dev/null", timeout=10)
print("Checkpoints:", stdout.read().decode().strip())

# Check latest training output containing save info
stdin, stdout, stderr = c.exec_command(
    "grep -i 'saved\|saving\|checkpoint' $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v3_*.log 2>/dev/null | head -1) 2>/dev/null | tail -5",
    timeout=10)
print("Save logs:", stdout.read().decode().strip())

c.close()
