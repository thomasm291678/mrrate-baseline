import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

NAS_D = "/mnt/nas1/disk07/jiaqigu_mrrate_ckpts"
SRC = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen"

# Check what NAS path is
stdin, o, e = c.exec_command(f"ls -la {NAS_D} 2>&1; file {NAS_D} 2>&1", timeout=10)
print("NAS check:", o.read().decode().strip())

# Fix: remove if file, then mkdir
c.exec_command(f"rm -f {NAS_D} 2>/dev/null; mkdir -p {NAS_D} 2>&1", timeout=10)

# Now copy
stdin, o, e = c.exec_command(
    f"cp -v {SRC}/step_*.pt {NAS_D}/ 2>&1 && "
    f"cp -v {SRC}/best_model.pt {NAS_D}/ 2>&1 && "
    f"ls -lh {NAS_D}/", timeout=600)
print("Result:", o.read().decode().strip())

c.close()
