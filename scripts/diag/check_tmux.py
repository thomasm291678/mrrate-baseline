import paramiko, time

print("等待 180 秒后检查...")
time.sleep(180)

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

s, o, e = c.exec_command("tmux capture-pane -t t1 -p -S -10 2>/dev/null", timeout=10)
print(o.read().decode())

s, o, e = c.exec_command("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep '^3,'", timeout=10)
print("GPU3:", o.read().decode().strip())

s, o, e = c.exec_command("tmux ls 2>/dev/null && echo 'tmux ALIVE'", timeout=10)
print(o.read().decode().strip())

c.close()
