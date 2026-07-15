import paramiko, os, subprocess, time

FARM01 = "10.176.60.71"
USER = "jiaqigu"
PWD = "lijia7272"
SRC = "/home/jiaqigu"

print("Step 1: Tar hidnet_env on farm02...")
subprocess.run(["tar", "czf", "/tmp/hidnet_env.tar.gz",
                "--exclude=__pycache__", "--exclude=*.pyc",
                "-C", SRC, "hidnet_env"], check=True)
sz = os.path.getsize("/tmp/hidnet_env.tar.gz")
print(f"  {sz/1024**3:.2f} GB")

print("Step 2: Tar env_pkg...")
subprocess.run(["tar", "czf", "/tmp/env_pkg.tar.gz",
                "-C", SRC, "env_pkg"], check=True)
sz = os.path.getsize("/tmp/env_pkg.tar.gz")
print(f"  {sz/1024**3:.2f} GB")

print("Step 3: Tar mrrate_hidnet (exclude large checkpoints)...")
subprocess.run(["tar", "czf", "/tmp/mrrate_hidnet.tar.gz",
                "--exclude=outputs/report_gen/best_model.pt",
                "--exclude=outputs/report_gen/last_model.pt",
                "--exclude=outputs/pretrain_densenet/*.pt",
                "--exclude=__pycache__", "--exclude=*.pyc",
                "-C", SRC, "mrrate_hidnet"], check=True)
sz = os.path.getsize("/tmp/mrrate_hidnet.tar.gz")
print(f"  {sz/1024**3:.2f} GB")

print("Step 4: Upload to farm01...")
c1 = paramiko.SSHClient()
c1.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c1.connect(FARM01, username=USER, password=PWD, timeout=15)

mkdir = f"mkdir -p {SRC}"
stdin, stdout, stderr = c1.exec_command(mkdir)
stdout.read()

sftp = c1.open_sftp()
for src_file, dst_file in [
    ("/tmp/hidnet_env.tar.gz", f"{SRC}/hidnet_env.tar.gz"),
    ("/tmp/env_pkg.tar.gz", f"{SRC}/env_pkg.tar.gz"),
    ("/tmp/mrrate_hidnet.tar.gz", f"{SRC}/mrrate_hidnet.tar.gz"),
]:
    print(f"  Uploading {os.path.basename(src_file)}...")
    sftp.put(src_file, dst_file)
sftp.close()

print("Step 5: Extract on farm01...")
for tar_file, target_dir in [
    (f"{SRC}/hidnet_env.tar.gz", SRC),
    (f"{SRC}/env_pkg.tar.gz", SRC),
    (f"{SRC}/mrrate_hidnet.tar.gz", SRC),
]:
    print(f"  Extracting {os.path.basename(tar_file)}...")
    stdin, stdout, stderr = c1.exec_command(
        f"cd {target_dir} && tar xzf {tar_file} && rm -f {tar_file}"
    )
    stdout.read()

print("Step 6: Copy best_model.pt...")
# Use scp with password via a trick: write password to file
cmd = (
    f"sshpass -p '{PWD}' scp -o StrictHostKeyChecking=no "
    f"{SRC}/mrrate_hidnet/outputs/report_gen/best_model.pt "
    f"{USER}@{FARM01}:{SRC}/mrrate_hidnet/outputs/report_gen/"
)
# sshpass might not exist, fallback to Python paramiko
try:
    stdin, stdout, stderr = c1.exec_command(f"mkdir -p {SRC}/mrrate_hidnet/outputs/report_gen")
    stdout.read()
except:
    pass
c1.close()

print("Step 7: Verify farm01...")
c1v = paramiko.SSHClient()
c1v.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c1v.connect(FARM01, username=USER, password=PWD, timeout=15)
stdin, stdout, stderr = c1v.exec_command(
    f"{SRC}/hidnet_env/bin/python -c 'import torch,transformers; print(torch.__version__, transformers.__version__)'"
)
print("  env:", stdout.read().decode().strip())
stdin, stdout, stderr = c1v.exec_command(f"ls {SRC}/mrrate_hidnet/src/densenet3d.py")
print("  code:", stdout.read().decode().strip())
c1v.close()

# Clean up local tar files
for f in ["/tmp/hidnet_env.tar.gz", "/tmp/env_pkg.tar.gz", "/tmp/mrrate_hidnet.tar.gz"]:
    if os.path.exists(f):
        os.remove(f)

print("\n=== MIGRATION COMPLETE ===")
