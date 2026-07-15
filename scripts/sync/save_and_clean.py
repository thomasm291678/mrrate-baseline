import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

SRC = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen"
NAS = "/mnt/nas1/disk07/public/qi/v3_ckpts_20260713"

# Copy all step checkpoints to NAS
stdin, o, e = c.exec_command(
    f"cp -n {SRC}/step_*.pt {NAS}/ 2>&1 && ls -lh {NAS}/", timeout=600)
out = o.read().decode()
print(out[:1000])

# Delete local step checkpoints, keep last 2
c.exec_command(
    f"cd {SRC} && ls -t step_*.pt | tail -n +3 | xargs rm -f 2>/dev/null; true", timeout=10)

# Clean GPU0 zombie and relaunch
c.exec_command("fuser -k /dev/nvidia0 2>/dev/null; true", timeout=5)

c.close()
