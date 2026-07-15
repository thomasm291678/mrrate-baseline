import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Check current training
stdin, o, e = c.exec_command("tail -3 $(ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_v3_*.log|head -1)", timeout=10)
print("Current:", o.read().decode().strip())

# Check available step checkpoints local + NAS
stdin, o, e = c.exec_command(
    "echo '=== Local ==='; ls -lh /home/jiaqigu/mrrate_hidnet/outputs/report_gen/step_*.pt 2>/dev/null; "
    "echo '=== NAS ==='; ls -lh /mnt/nas1/disk07/public/qi/v3_ckpts_20260713/step_*.pt 2>/dev/null",
    timeout=10)
print("\n" + o.read().decode().strip())

# Check which step_*.pt files were lost
stdin, o, e = c.exec_command(
    "echo '=== NAS full ==='; ls -lh /mnt/nas1/disk07/public/qi/v3_ckpts_20260713/", timeout=10)
print("\n" + o.read().decode().strip())

c.close()
