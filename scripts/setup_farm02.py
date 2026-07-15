import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272")

# Find ALL phase1 checkpoints
_, o, _ = c.exec_command("find /home/jiaqigu -name '*phase1*.pt' -type f 2>&1; find /mnt/nas1 -name '*phase1*.pt' -type f -user jiaqigu 2>&1 | head -10")
print("Phase1 ckpts:", o.read().decode())

# Check farm02 for Phase 1 checkpoints from the training we killed
c.close()
