import paramiko, time

for name, ip in [("farm04", "10.176.60.71"), ("farm05", "10.176.60.72")]:
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(ip, username="jiaqigu", password="lijia7272", timeout=10)
        c.exec_command("pkill -9 -u jiaqigu 2>/dev/null; true", timeout=5)
        c.close()
        print(f"{name}: killed")
    except:
        print(f"{name}: down")
