import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Fix: reinstall torchvision matching torch 2.5.1+cu124
script = '''#!/bin/bash
PIP=/home/jiaqigu/hidnet_env/bin/pip
$PIP uninstall torchvision -y 2>/dev/null
$PIP install torchvision --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -5
echo "---"
/home/jiaqigu/hidnet_env/bin/python -c "
import torch; print('torch', torch.__version__)
import torchvision; print('torchvision', torchvision.__version__)
from transformers import AutoModelForCausalLM; print('transformers OK')
from peft import LoraConfig; print('peft OK')
print('ALL OK')
"
'''

c.exec_command(f"cat > /home/jiaqigu/mrrate_hidnet/fix_tv.sh << 'ENDF'\n{script}\nENDF", timeout=5)
c.exec_command("chmod +x /home/jiaqigu/mrrate_hidnet/fix_tv.sh", timeout=5)
c.exec_command("nohup bash /home/jiaqigu/mrrate_hidnet/fix_tv.sh > /home/jiaqigu/mrrate_hidnet/outputs/report_gen/fix_tv.log 2>&1 &", timeout=5)
print("fix_tv.sh running in background")
print("Check: outputs/report_gen/fix_tv.log")
c.close()
