import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=30)

stdin, stdout, stderr = c.exec_command(
    'head -5 $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_*.log | head -1) && '
    'echo "=== GPU ===" && nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | grep "^6" && '
    'echo "=== CPU ===" && top -bn1 -u jiaqigu | head -5'
)
print(stdout.read().decode(errors="replace"))

# Also check latest log for timing
stdin, stdout, stderr = c.exec_command(
    'grep -E "\\[E001 S[0-9]+\\]" /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_*.log | '
    'awk \'{print $1,$2,$3,$4}\' | tail -5 && '
    'echo "---" && '
    'grep -E "\\[E001 S[0-9]+\\]" /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_*.log | '
    'head -2'
)
print(stdout.read().decode(errors="replace"))

c.close()
