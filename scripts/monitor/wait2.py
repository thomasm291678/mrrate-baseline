import paramiko, time
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=30)

for i in range(6):
    time.sleep(60)
    cmds = [
        'tail -2 $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_*.log | head -1) 2>/dev/null',
        'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | grep "^6"',
        'ps -u jiaqigu | grep python | wc -l',
    ]
    stdin, stdout, stderr = c.exec_command("; ".join(cmds), timeout=30)
    out = stdout.read().decode(errors="replace").strip()
    print(f"[{i+1}] GPU mem={out.splitlines()[-2].strip() if len(out.splitlines())>1 else '?'}")
    if "[E001" in out:
        print("RUNNING!")
        print(out)
        break
c.close()
