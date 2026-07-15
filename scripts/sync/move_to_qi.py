import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

SRC = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen"
QI = "/mnt/nas1/disk07/public/qi"
NEW = f"{QI}/v3_ckpts_20260713"

# Remove the separate dir
c.exec_command(f"rm -rf /mnt/nas1/disk07/public/jiaqigu_ckpts", timeout=10)

# Create new dir under qi
c.exec_command(f"mkdir -p {NEW}", timeout=10)

# Copy
stdin, o, e = c.exec_command(
    f"cp -v {SRC}/step_*.pt {NEW}/ 2>&1 && "
    f"cp -v {SRC}/best_model.pt {NEW}/ 2>&1 && "
    f"ls -lh {NEW}/",
    timeout=600)
print(o.read().decode())

# Clean local step checkpoints (keep only for current training)
c.exec_command(f"rm -f {SRC}/step_*.pt && echo 'cleaned step_*.pt' && df -h {SRC}/.. | tail -1", timeout=10)

c.close()
