import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Check if GPU6 process is alive
s, o, e = c.exec_command("ps -u jiaqigu | grep 'train.py' | grep -v grep", timeout=10)
procs = o.read().decode().strip()
print("Processes:")
for line in procs.split("\n"):
    print(f"  {line[:150]}")

# Check latest log with real content
s, o, e = c.exec_command(
    "ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_gpu6_*.log | head -1", timeout=10)
log = o.read().decode().strip()

s, o, e = c.exec_command(f"tail -10 {log}", timeout=10)
lines = o.read().decode().strip()
print(f"\nLatest log ({log.split('/')[-1]}):")
print(lines)

# Check if step advances or stuck
s, o, e = c.exec_command(f"grep ' S0' {log} | tail -3", timeout=10)
print("\nLast 3 step lines:")
print(o.read().decode().strip())

c.close()
