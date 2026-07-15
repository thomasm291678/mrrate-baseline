import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# GPU1 is new, GPU0 is old
s, o, e = c.exec_command(
    "echo '=== GPU0 ==='; tail -3 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_20260713_075047.log; "
    "echo '=== GPU2 (v3) ==='; tail -3 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v3_20260713_085119.log; "
    "echo '=== PROC ==='; ps aux | grep train.py | grep -v grep",
    timeout=15)
print(o.read().decode())

c.close()
