import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

s, o, e = c.exec_command(
    "cat /home/jiaqigu/mrrate_hidnet/outputs/report_gen/v4_all3_20260713_162000.log 2>/dev/null",
    timeout=10)
print("=== v4_all3 log ===")
print(o.read().decode())
print("[EOF]")

s, o, e = c.exec_command(
    "cat /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v4_t1.log 2>/dev/null | tail -20",
    timeout=10)
print("\n=== t1 log ===")
print(o.read().decode())

c.close()
