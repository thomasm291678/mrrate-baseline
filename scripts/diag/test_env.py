import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=15)

# Write test script on server
test = """import torch
print("torch", torch.__version__, torch.cuda.is_available())
import torch.nn as nn
print("nn OK")
from transformers import AutoModelForCausalLM
print("transformers OK")
from peft import LoraConfig
print("peft OK")
print("ALL OK")
"""
c.exec_command(f"cat > /tmp/test_env.py << 'EOF'\n{test}\nEOF", timeout=5)

stdin, stdout, stderr = c.exec_command(
    "/home/jiaqigu/hidnet_env/bin/python /tmp/test_env.py 2>&1", timeout=15)
print(stdout.read().decode() + stderr.read().decode())
c.close()
