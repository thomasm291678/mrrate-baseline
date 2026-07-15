import paramiko

for host in ["10.176.60.71", "10.154.32.115"]:
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(host, username="jiaqigu", password="lijia7272", timeout=15)
        stdin, stdout, stderr = c.exec_command("echo OK", timeout=5)
        print(f"{host}: {stdout.read().decode().strip()}")
        stdin, stdout, stderr = c.exec_command(
            "cd /home/jiaqigu/mrrate_hidnet && git clone https://github.com/BAAI-DCAI/M3D.git research_m3d 2>&1",
            timeout=60)
        out = stdout.read().decode(errors="replace") + stderr.read().decode(errors="replace")
        print(out[:300])
        c.close()
        break
    except Exception as e:
        print(f"{host}: FAIL - {e}")
