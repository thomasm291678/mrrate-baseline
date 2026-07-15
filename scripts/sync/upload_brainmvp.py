import paramiko, time

src = r"C:\Users\HP\Downloads\BrainMVP_uniformer.pt"
dst_path = "/home/jiaqigu/mrrate_hidnet/outputs/report_gen/BrainMVP_uniformer.pt"

for name, ip in [("farm04", "10.176.60.71"), ("farm05", "10.176.60.72")]:
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(ip, username="jiaqigu", password="lijia7272", timeout=15)
        sftp = c.open_sftp()
        print(f"{name}: uploading 1.13GB...")
        sftp.put(src, dst_path)
        sftp.close()
        
        # verify
        s, o, e = c.exec_command(f"ls -lh {dst_path} && md5sum {dst_path}", timeout=30)
        print(f"{name}: {o.read().decode().strip()[:120]}")
        c.close()
    except Exception as ex:
        print(f"{name}: {ex}")

print("Done")
