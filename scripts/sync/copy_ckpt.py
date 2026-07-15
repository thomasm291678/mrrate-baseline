import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

LOCAL = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen"
NAS_CKPT = "/mnt/nas1/disk07/public/qi/v3_ckpts_20260713/step_001600.pt"

# Background copy
c.exec_command(f"nohup cp {NAS_CKPT} {LOCAL}/step_001600.pt &", timeout=5)
print("Copying step_001600.pt from NAS in background...")

# Check if copy done
import time
time.sleep(60)

stdin, o, e = c.exec_command(f"ls -lh {LOCAL}/step_001600.pt 2>/dev/null && echo 'DONE' || echo 'COPYING'", timeout=10)
print(o.read().decode().strip())

c.close()
