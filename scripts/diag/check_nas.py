import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Check NAS permissions
stdin, o, e = c.exec_command(
    "ls -la /mnt/nas1/disk07/ | head -10 && echo '---' && "
    "touch /mnt/nas1/disk07/test_write 2>&1 && echo 'WRITE OK' || echo 'WRITE FAILED' && "
    "rm -f /mnt/nas1/disk07/test_write 2>/dev/null && "
    "mkdir /mnt/nas1/disk07/test_dir 2>&1 && echo 'MKDIR OK' && rmdir /mnt/nas1/disk07/test_dir || echo 'MKDIR FAILED'",
    timeout=15)
print(o.read().decode())

# Check who owns the NAS
stdin, o, e = c.exec_command("whoami; id; mount | grep nas1", timeout=10)
print(o.read().decode())

c.close()
