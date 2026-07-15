import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=15)

base = "/mnt/nas1/disk07/public/shushangjiang /evaluation.v1"

# Read full extract_labels_keyword.py
s, o, e = c.exec_command(f"cat '{base}/extract_labels_keyword.py'")
print("=== FULL extract_labels_keyword.py ===")
print(o.read().decode())

c.close()
