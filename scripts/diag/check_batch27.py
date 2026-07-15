import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=30)

stdin, stdout, stderr = c.exec_command(
    "ls /mnt/nas1/disk07/public/mr_data/MR-RATE/mri/ | sort", timeout=30)
batches = stdout.read().decode(errors="replace").strip().split("\n")
print(f"Total batches: {len(batches)}")
print(f"Last 5: {batches[-5:]}")

stdin, stdout, stderr = c.exec_command(
    "wc -l /mnt/nas1/disk07/public/mr_data/MR-RATE/splits.csv", timeout=15)
print(f"splits.csv lines: {stdout.read().decode().strip()}")

stdin, stdout, stderr = c.exec_command(
    "grep -c 'batch27' /mnt/nas1/disk07/public/mr_data/MR-RATE/splits.csv 2>/dev/null || echo '0'",
    timeout=15)
print(f"batch27 in splits: {stdout.read().decode().strip()}")

c.close()
