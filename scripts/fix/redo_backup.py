import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

SRC = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen"
NAS = "/mnt/nas1/disk07/jiaqigu_mrrate_ckpts"

# Create dir with proper permissions
stdin, o, e = c.exec_command(
    f"sudo mkdir -p {NAS} && sudo chown jiaqigu:10000 {NAS}", timeout=10)
print("Mkdir:", o.read().decode().strip() + e.read().decode().strip())

# List local ckpts
stdin, o, e = c.exec_command(f"ls -lh {SRC}/step_*.pt 2>/dev/null", timeout=10)
print("Local ckpts:", o.read().decode().strip()[:500])

# Copy remaining
stdin, o, e = c.exec_command(
    f"cp -v {SRC}/step_*.pt {NAS}/ 2>&1 && cp -v {SRC}/best_model.pt {NAS}/ 2>&1 && ls -lh {NAS}/",
    timeout=600)
print("Copy:", o.read().decode().strip())

c.close()
