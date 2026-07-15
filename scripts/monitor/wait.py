import paramiko, time
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=30)

for i in range(5):
    time.sleep(55)
    cmds = [
        'tail -3 $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_*.log | head -1)',
        'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep "^6"',
    ]
    stdin, stdout, stderr = c.exec_command("; ".join(cmds), timeout=30)
    out = stdout.read().decode(errors="replace").strip()
    print(f"[{i+1}] {out}")
    if "E001" in out:
        print("RUNNING!")
        break
c.close()
