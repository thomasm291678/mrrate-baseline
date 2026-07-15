import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

s, o, e = c.exec_command("grep ',train$' /mnt/nas1/disk07/public/mr_data/MR-RATE/splits.csv | grep '^batch27,' | wc -l", timeout=10)
print("batch27 train:", o.read().decode().strip())

s, o, e = c.exec_command("grep ',train$' /mnt/nas1/disk07/public/mr_data/MR-RATE/splits.csv | grep '^batch27,' | head -3", timeout=10)
print(o.read().decode().strip())

s, o, e = c.exec_command("grep ',val$' /mnt/nas1/disk07/public/mr_data/MR-RATE/splits.csv | grep '^batch27,' | wc -l", timeout=10)
print("batch27 val:", o.read().decode().strip())

c.close()
