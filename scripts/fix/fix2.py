import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Quick fix: remove torchaudio, reinstall peft to fix compat
script = '''#!/bin/bash
PIP=/home/jiaqigu/hidnet_env/bin/pip
PY=/home/jiaqigu/hidnet_env/bin/python
set -x
$PIP uninstall torchaudio -y 2>/dev/null
$PIP install peft --upgrade 2>&1 | tail -3
$PY -c "import torch; from peft import LoraConfig; print('PEft OK')" 2>&1
'''

c.exec_command(f"cat > /home/jiaqigu/mrrate_hidnet/fix2.sh << 'EOF'\n{script}\nEOF", timeout=5)
c.exec_command("chmod +x /home/jiaqigu/mrrate_hidnet/fix2.sh", timeout=5)
c.exec_command("nohup bash /home/jiaqigu/mrrate_hidnet/fix2.sh > /home/jiaqigu/mrrate_hidnet/outputs/report_gen/fix2.log 2>&1 &", timeout=5)
print("fix2.sh running...")
c.close()
