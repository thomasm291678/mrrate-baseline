import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.70", username="jiaqigu", password="lijia7272", timeout=15)

# Find python
s, o, e = c.exec_command("which python3 && python3 --version", timeout=10)
print("python3:", o.read().decode().strip())

s, o, e = c.exec_command("ls /home/jiaqigu/ | head -20", timeout=10)
print("home:", o.read().decode().strip())

# Check if we can install from farm04 using scp with ssh key
s, o, e = c.exec_command("ssh-keyscan -H 10.176.60.71 >> ~/.ssh/known_hosts 2>/dev/null; cat ~/.ssh/known_hosts | wc -l", timeout=10)
print("known_hosts:", o.read().decode().strip())

# Actually try to copy the whole env
s, o, e = c.exec_command(
    "rsync -avP --exclude='*.pyc' --exclude='__pycache__' --exclude='.git' 10.176.60.71:/home/jiaqigu/hidnet_env/ /home/jiaqigu/hidnet_env/ 2>&1 | tail -5",
    timeout=30)
print("rsync:", o.read().decode().strip())

c.close()
