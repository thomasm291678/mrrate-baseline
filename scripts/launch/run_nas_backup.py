import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.154.32.115", username="jiaqigu", password="lijia7272", timeout=30)

cmds = [
    'echo "=== NAS Backup ==="',
    'cd /home/jiaqigu/mrrate_hidnet',
    'python3 nas_backup.py 2>&1',
]
cmd_str = "; ".join(cmds)
print(f"Running on farm02...")
stdin, stdout, stderr = c.exec_command(cmd_str, timeout=600)
out = stdout.read().decode(errors="replace")
err = stderr.read().decode(errors="replace")
print(out)
if err:
    print(f"STDERR: {err[:500]}")
c.close()
print("DONE")
