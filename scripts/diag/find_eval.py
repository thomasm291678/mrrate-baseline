import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=15)

print("=== Searching NAS for evaluation ===")
s, o, e = c.exec_command("find /mnt/nas1/disk07/public -maxdepth 4 -type d -iname '*eval*' 2>/dev/null | head -20")
print(o.read().decode().strip())

print("\n=== Searching home for evaluation ===")
s, o, e = c.exec_command("find /home/jiaqigu -maxdepth 4 -type d -iname '*eval*' 2>/dev/null | head -20")
print(o.read().decode().strip())

print("\n=== Phase 2 checkpoint ===")
s, o, e = c.exec_command("ls -lh /home/jiaqigu/mrrate_hidnet/outputs/report_gen/phase2_latest.pt 2>/dev/null")
print(o.read().decode().strip())

c.close()
