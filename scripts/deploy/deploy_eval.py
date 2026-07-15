#!/usr/bin/env python3
"""Copy JSS evaluation.v1 files from NAS to local, then upload merged evaluation to all targets"""
import paramiko, shutil, os, json
from pathlib import Path

LOCAL_DIR = Path(r"C:\Users\HP\Documents\5555")
EVAL_DIR = LOCAL_DIR / "evaluation"
NAS_EVAL_BASE = "/mnt/nas1/disk07/public/jiaqigu/evaluation"
JSS_PATH = "/mnt/nas1/disk07/public/shushangjiang /evaluation.v1"

# ========== Step 1: Copy JSS files from NAS to local ==========
print("=== Step 1: Copy JSS evaluation.v1 from NAS ===")
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=15)

# Create local dirs
(EVAL_DIR / "reference" / "evaluation_v1_jss").mkdir(parents=True, exist_ok=True)
(EVAL_DIR / "v2").mkdir(parents=True, exist_ok=True)

sftp = c.open_sftp()
jss_files = sftp.listdir(JSS_PATH)
for fname in jss_files:
    if fname == "__pycache__":
        continue
    remote_path = f"{JSS_PATH}/{fname}"
    local_path = EVAL_DIR / "reference" / "evaluation_v1_jss" / fname
    try:
        sftp.get(remote_path, str(local_path))
        print(f"  Downloaded: {fname}")
    except IsADirectoryError:
        pass
    except Exception as e:
        print(f"  Skip {fname}: {e}")
sftp.close()
print("  Done.\n")

# ========== Step 2: Copy evaluation_v2.py to local eval dir ==========
shutil.copy2(LOCAL_DIR / "evaluation_v2.py", EVAL_DIR / "v2" / "evaluation_v2.py")
print("=== Step 2: evaluation_v2.py → local evaluation/v2/ ===\n")

# ========== Step 3: Upload to NAS ==========
print("=== Step 3: Upload to NAS ===")
c.exec_command(f"mkdir -p '{NAS_EVAL_BASE}/reference/evaluation_v1_jss'")
c.exec_command(f"mkdir -p '{NAS_EVAL_BASE}/v2'")

sftp = c.open_sftp()

# Upload JSS reference
for f in (EVAL_DIR / "reference" / "evaluation_v1_jss").iterdir():
    remote = f"{NAS_EVAL_BASE}/reference/evaluation_v1_jss/{f.name}"
    sftp.put(str(f), remote)
    print(f"  NAS upload: reference/evaluation_v1_jss/{f.name}")

# Upload v2
remote_v2 = f"{NAS_EVAL_BASE}/v2/evaluation_v2.py"
sftp.put(str(EVAL_DIR / "v2" / "evaluation_v2.py"), remote_v2)
print(f"  NAS upload: v2/evaluation_v2.py")

# Upload unified_eval.py as legacy reference
remote_legacy = f"{NAS_EVAL_BASE}/v1_legacy/unified_eval.py"
c.exec_command(f"mkdir -p '{NAS_EVAL_BASE}/v1_legacy'")
sftp.put(str(LOCAL_DIR / "unified_eval.py"), remote_legacy)
print(f"  NAS upload: v1_legacy/unified_eval.py")

sftp.close()
print("  Done.\n")

# ========== Step 4: Verify NAS ==========
print("=== Step 4: Verify NAS structure ===")
s, o, e = c.exec_command(f"find '{NAS_EVAL_BASE}' -type f | sort")
print(o.read().decode())

c.close()
print("\nAll done!")
