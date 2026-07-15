import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Check public/ dir
stdin, o, e = c.exec_command(
    "ls -la /mnt/nas1/disk07/public/ | head -20 && echo '---' && "
    "df -h /mnt/nas1/disk07/", timeout=10)
print(o.read().decode())

# Try mkdir in public
stdin, o, e = c.exec_command(
    "mkdir /mnt/nas1/disk07/public/jiaqigu_ckpts 2>&1 && echo 'OK' || echo 'FAIL'",
    timeout=10)
print("Mkdir:", o.read().decode().strip())

c.close()
