import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=10)

c.exec_command("pkill -9 -f train_v5; sleep 1")

sf = c.open_sftp()
with open(r"C:\Users\HP\Documents\5555\run_phase2.sh", "rb") as f:
    sf.putfo(f, "/home/jiaqigu/mrrate_hidnet/run_phase2.sh")
sf.close()
print("uploaded")

c.exec_command("chmod +x /home/jiaqigu/mrrate_hidnet/run_phase2.sh")

transport = c.get_transport()
channel = transport.open_session()
channel.exec_command("cd /home/jiaqigu/mrrate_hidnet && nohup bash run_phase2.sh > /dev/null 2>&1 < /dev/null & echo OK")
channel.close()
print("launched")

c.close()
