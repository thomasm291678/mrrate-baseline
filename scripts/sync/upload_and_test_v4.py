import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Upload new files
sftp = c.open_sftp()
sftp.put(r"C:\Users\HP\Documents\5555\uniformer_blocks.py", 
         "/home/jiaqigu/mrrate_hidnet/uniformer_blocks.py")
sftp.put(r"C:\Users\HP\Documents\5555\encoder_v4.py", 
         "/home/jiaqigu/mrrate_hidnet/encoder_v4.py")
sftp.put(r"C:\Users\HP\Documents\5555\train_v4.py", 
         "/home/jiaqigu/mrrate_hidnet/scripts/train_v4.py")
sftp.close()
print("Uploaded v4 files")

# Test encoder_v4
stdin, o, e = c.exec_command(
    "cd /home/jiaqigu/mrrate_hidnet && "
    "python -c \""
    "import torch; from encoder_v4 import UniFormerEncoder, ReportingModelV4; "
    "x=torch.randn(2,1,128,128,128); enc=UniFormerEncoder(); out=enc(x); "
    "print(f'Encoder: {tuple(x.shape)} -> {tuple(out.shape)}'); "
    "print(f'UniFormer params: {sum(p.numel() for p in enc.uniformer.parameters()):,}'); "
    "print(f'Total encoder params: {sum(p.numel() for p in enc.parameters()):,}'); "
    "has=torch.tensor([True,True]); d=torch.zeros(2,1,128,128,128); "
    "m=ReportingModelV4(); vt=m(d,d,d,has,has,has); "
    "print(f'ReportingModelV4 output: {tuple(vt.shape)}'); "
    "print(f'Total model params: {sum(p.numel() for p in m.parameters()):,}')\"",
    timeout=60)
print(o.read().decode())

c.close()
