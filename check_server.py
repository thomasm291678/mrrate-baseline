import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=10)

_, o, _ = c.exec_command("head -20 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/phase2.log")
print("HEAD:")
print(o.read().decode())

_, o, _ = c.exec_command("cat /home/jiaqigu/mrrate_hidnet/outputs/report_gen/phase2.log | grep 'Epoch\|Phase\|Finished\|Train:\|Val:'")
print("SUMMARY:")
print(o.read().decode())

_, o, _ = c.exec_command("head -5 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/phase2.log; echo '---'; grep 'avg_loss' /home/jiaqigu/mrrate_hidnet/outputs/report_gen/phase2.log | head -5; echo '---'; tail -3 /home/jiaqigu/mrrate_hidnet/outputs/report_gen/phase2.log")
print("DETAIL:")
print(o.read().decode())

c.close()
