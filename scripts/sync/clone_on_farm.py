import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=30)

cmds = [
    'cd /home/jiaqigu/mrrate_hidnet && git clone https://github.com/BAAI-DCAI/M3D.git research_m3d 2>&1',
    'cd /home/jiaqigu/mrrate_hidnet && git clone https://github.com/mk-runner/Awesome-Radiology-Report-Generation.git research_awesome 2>&1',
]
for cmd in cmds:
    stdin, stdout, stderr = c.exec_command(cmd, timeout=60)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    print(f"CMD: {cmd[:80]}...")
    print(out[:200] or err[:200])

# Verify
stdin, stdout, stderr = c.exec_command(
    "ls /home/jiaqigu/mrrate_hidnet/research_m3d/LaMed/ 2>/dev/null && "
    "ls /home/jiaqigu/mrrate_hidnet/research_m3d/LaMed/src/ 2>/dev/null && "
    "wc -l /home/jiaqigu/mrrate_hidnet/research_awesome/README.md 2>/dev/null")
print(stdout.read().decode(errors="replace"))

c.close()
