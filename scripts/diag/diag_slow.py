import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.71", username="jiaqigu", password="lijia7272", timeout=30)

cmds = [
    'echo "=== GPU utilization over time ==="',
    'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | grep "^6"',
    'echo "=== Dataset info ==="',
    'echo "samples: 88985, batch=4, steps_per_epoch=22246"',
    'echo "=== Current speed ==="',
    'grep -E "\\[E001" /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_20260712_195752.log | tail -3',
    'echo "=== IO wait ==="',
    'top -bn2 -d 2 | grep "Cpu" | tail -1',
    'echo "=== NAS latency ==="',
    'timeout 2 dd if=/mnt/nas1/disk07/public/mr_data/MR-RATE/splits.csv of=/dev/null bs=4k 2>&1 | tail -1',
    'echo "=== DISK ==="',
    'df -h /mnt/nas1/disk07/',
    'echo "=== Local disk ==="',
    'df -h /home/jiaqigu/',
]

stdin, stdout, stderr = c.exec_command("; ".join(cmds), timeout=60)
print(stdout.read().decode(errors="replace"))
c.close()
