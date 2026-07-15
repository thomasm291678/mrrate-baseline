import paramiko

for name, ip in [("farm04 GPU6", "10.176.60.71"), ("farm05 GPU0", "10.176.60.72")]:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(ip, username="jiaqigu", password="lijia7272", timeout=15)
    
    # Get latest log
    s, o, e = c.exec_command(
        "ls -t /home/jiaqigu/mrrate_hidnet/outputs/report_gen/train_*.log "
        "2>/dev/null | head -1", timeout=10)
    log = o.read().decode().strip()
    
    s, o, e = c.exec_command(f"tail -5 {log} 2>/dev/null", timeout=10)
    lines = o.read().decode().strip()
    
    s, o, e = c.exec_command(
        "nvidia-smi --query-gpu=index,memory.used,utilization.gpu "
        "--format=csv,noheader 2>/dev/null | head -2", timeout=10)
    gpu = o.read().decode().strip()
    
    print(f"=== {name} ===")
    print(f"Log: {log}")
    print(lines)
    print(f"GPU: {gpu}")
    print()
    c.close()
