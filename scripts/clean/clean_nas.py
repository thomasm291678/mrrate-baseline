import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.154.32.115", username="jiaqigu", password="lijia7272", timeout=20)

stdin, stdout, stderr = c.exec_command(
    "rmdir /mnt/nas1/disk07/public/qi/checkpoints 2>/dev/null && echo DELETED || echo NOT_EMPTY_OR_MISSING")
print(stdout.read().decode().strip())

# verify
stdin, stdout, stderr = c.exec_command("ls -la /mnt/nas1/disk07/public/qi/")
print(stdout.read().decode())

c.close()
