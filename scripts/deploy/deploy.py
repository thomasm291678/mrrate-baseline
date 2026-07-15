import paramiko, sys, time

HOST = "10.176.60.71"
USER = "jiaqigu"
PWD = "lijia7272"
REMOTE = "/home/jiaqigu/mrrate_hidnet"

FILES = [
    ("encoder.py", f"{REMOTE}/src/encoder.py"),
    ("train.py", f"{REMOTE}/scripts/train.py"),
    ("run.sh", f"{REMOTE}/run.sh"),
    ("watchdog.sh", f"{REMOTE}/watchdog.sh"),
]


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PWD, timeout=15)

    # Upload source files
    sftp = c.open_sftp()
    for local, remote in FILES:
        print(f"  {local} -> {remote}")
        sftp.put(local, remote)
    sftp.close()
    c.exec_command(f"chmod +x {REMOTE}/run.sh {REMOTE}/watchdog.sh")
    print("Files uploaded.")

    # Check env
    stdin, stdout, stderr = c.exec_command(
        f"ls {REMOTE}/hidnet_env/bin/python && echo ENV_OK || echo NO_ENV")
    env_status = stdout.read().decode().strip()
    print(f"env: {env_status[:50]}")

    # Check GPU6
    stdin, stdout, stderr = c.exec_command(
        "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep -A1 '^6'")
    gpu6 = stdout.read().decode().strip()
    print(f"GPU6: {gpu6}")

    # Check best_model.pt
    stdin, stdout, stderr = c.exec_command(
        f"ls -lh {REMOTE}/outputs/report_gen/best_model.pt 2>/dev/null || echo NO_MODEL")
    print(stdout.read().decode().strip())

    # Dry-run
    print("\n=== Dry-run (10 samples) ===")
    stdin, stdout, stderr = c.exec_command(
        f"cd {REMOTE} && CUDA_VISIBLE_DEVICES=6 "
        f"/home/jiaqigu/hidnet_env/bin/python scripts/train.py "
        f"--v1_ckpt outputs/report_gen/best_model.pt "
        f"--max_samples 10 --log_interval 1 --epochs 1 "
        f"--grid 2 --vit_dim 512 --vit_heads 8 --vit_depth 2 "
        f"--batch_size 2 --ga_steps 2 --use_amp 2>&1", timeout=600)
    out = stdout.read().decode()
    err = stderr.read().decode()
    print(out[-3000:] if len(out) > 3000 else out)
    if err:
        print("STDERR:", err[-300:])
    c.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
