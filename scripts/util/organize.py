import os, shutil

ROOT = r"C:\Users\HP\Documents\5555"

# Round 2: catch unknowns
EXTRA_MOVES = {
    # diag
    "check16.py": "scripts/diag",
    "diag.py": "scripts/diag",
    "diag2.py": "scripts/diag",
    "deep_diag.py": "scripts/diag",
    # launch
    "bs1.py": "scripts/launch",
    "bs3.py": "scripts/launch",
    "uni_bs1.py": "scripts/launch",
    "start.py": "scripts/launch",
    "nocompile.py": "scripts/launch",
    "retry_farms.py": "scripts/launch",
    "py_farm03.py": "scripts/launch",
    "v5_fix.py": "scripts/launch",
    "v5_noamp.py": "scripts/launch",
    "tmux_t1.py": "scripts/launch",
    "tmux_train.py": "scripts/launch",
    # fix
    "fix2.py": "scripts/fix",
    "restart2.py": "scripts/fix",
    "restart3.py": "scripts/fix",
    # monitor
    "monitor.py": "scripts/monitor",
    "watch.py": "scripts/monitor",
    "last_log.py": "scripts/monitor",
    "status.py": "scripts/monitor",
    "status2.py": "scripts/monitor",
    "wait.py": "scripts/monitor",
    "wait2.py": "scripts/monitor",
    "wait_farm05.py": "scripts/monitor",
    "wait_t2.py": "scripts/monitor",
    "watchdog_daemon.py": "scripts/monitor",
    "quick_fix.py": "scripts/monitor",
    # test
    "test_save.py": "scripts/diag",
    "test_save2.py": "scripts/diag",
    "test_v4.py": "scripts/diag",
    "test_v4_proj.py": "scripts/diag",
    "test_v5.py": "scripts/diag",
    "test_v5_srv.py": "scripts/diag",
    # util
    "scan.py": "scripts/util",
    "plot_loss.py": "scripts/util",
    "update_readme.py": "scripts/util",
    "organize.py": "scripts/util",
}

for fname, cat in EXTRA_MOVES.items():
    src = os.path.join(ROOT, fname)
    if not os.path.exists(src):
        print(f"  MISSING: {fname}")
        continue
    dst_dir = os.path.join(ROOT, cat)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, fname)
    if os.path.exists(dst):
        print(f"  SKIP (exists): {fname}")
        continue
    shutil.move(src, dst)
    print(f"  OK: {fname} -> {cat}/")

# delete unnecessary junk
JUNK = [
    "dual_monitor.bat", "dual_monitor.log", "training_status.log",
    "check_monitor.sh", "run.sh", "run_radbert.sh", "watchdog.sh",
]
for f in JUNK:
    fp = os.path.join(ROOT, f)
    if os.path.exists(fp):
        os.remove(fp)
        print(f"  DEL: {f}")

print("\nDone round 2.")
