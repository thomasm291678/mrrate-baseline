import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Write background repair script
script = '''#!/bin/bash
cd /home/jiaqigu/mrrate_hidnet
PIP=/home/jiaqigu/hidnet_env/bin/pip
PY=/home/jiaqigu/hidnet_env/bin/python
LOGFILE=outputs/report_gen/repair.log

{
echo "=== $(date) Starting repair ==="
$PIP install evaluate scikit-learn rouge_score 2>&1 | tail -3
echo "=== $(date) evaluate done ==="

$PIP uninstall torch torchvision -y 2>&1 | tail -3
$PIP install torch torchvision 2>&1 | tail -5
echo "=== $(date) torch reinstalled ==="

$PY -c "import torch; print('torch OK', torch.cuda.is_available())"
$PY -c "import evaluate; print('evaluate OK')"

echo "=== $(date) Repair complete ==="
} >> $LOGFILE 2>&1
echo "REPAIR DONE" >> $LOGFILE
'''

c.exec_command(f"cat > /home/jiaqigu/mrrate_hidnet/repair.sh << 'ENDOFFILE'\n{script}\nENDOFFILE", timeout=5)
c.exec_command("chmod +x /home/jiaqigu/mrrate_hidnet/repair.sh", timeout=5)
c.exec_command("nohup bash /home/jiaqigu/mrrate_hidnet/repair.sh &", timeout=5)
print("repair.sh running in background")
print("Check: cat outputs/report_gen/repair.log")
print("Will take ~5-10 min (2GB torch download + evaluate)")
c.close()
