import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

s, o, e = c.exec_command(
    "tmux capture-pane -t t1 -p -S -15 2>/dev/null", timeout=10)
print("=== tmux t1 last 15 lines ===")
print(o.read().decode())

s, o, e = c.exec_command(
    "wc -l /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_t1_20260713_163707.log", timeout=10)
print(f"\nLog lines: {o.read().decode().strip()}")

s, o, e = c.exec_command(
    "ls -lh /home/jiaqigu/mrrate_hidnet/outputs/report_gen/latest_step.pt 2>/dev/null || echo 'no latest_step.pt'", timeout=10)
print(o.read().decode().strip())

s, o, e = c.exec_command("ps -u jiaqigu | grep 'train_v4' | grep -v grep", timeout=10)
print(f"\nProcesses:\n{o.read().decode().strip()}")

c.close()
