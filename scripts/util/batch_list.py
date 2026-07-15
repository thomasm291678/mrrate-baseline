import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

s, o, e = c.exec_command("ls /mnt/nas1/disk07/public/mr_data/MR-RATE/mri/ 2>/dev/null | head -30", timeout=10)
print("Batches:\n" + o.read().decode().strip())

s, o, e = c.exec_command("cat /mnt/nas1/disk07/public/mr_data/MR-RATE/splits.csv | head -5", timeout=10)
print("\nSplits head:")
print(o.read().decode().strip())

s, o, e = c.exec_command("cat /mnt/nas1/disk07/public/mr_data/MR-RATE/splits.csv | cut -d',' -f1 | sort -u", timeout=10)
print("\nBatch IDs in splits:", o.read().decode().strip())

c.close()
