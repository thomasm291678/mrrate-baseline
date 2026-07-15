#!/usr/bin/env python3
"""
NAS 一键备份脚本
运行位置: farm02 (10.154.32.115)
NAS 路径: /mnt/nas1/disk07/public/qi/
映射自:   \\10.154.32.108\shared  →  Z:\nas1\disk07\public\qi
本脚本将 farm02 上所有 MR-RATE 相关文件复制到 NAS 指定目录。
"""

import os
import shutil
import subprocess
import time
import sys

NAS_ROOT = "/mnt/nas1/disk07/public/qi"
SRC = "/home/jiaqigu/mrrate_hidnet"
SRC_EXTRA = "/home/jiaqigu/hidnet_env"

# NAS 目录结构:
#   qi/
#   ├── README.md              ← 完整说明文档
#   ├── weights/               ← 模型权重
#   │   ├── best_model.pt      ← V1 最终权重 (DenseNet3D + Qwen2.5-3B LoRA)
#   │   └── last_model.pt      ← V1 最后一个 epoch 权重
#   ├── code/                  ← 核心代码
#   │   ├── encoder.py         ← V3 Spark3D 多尺度 CNN+ViT 编码器
#   │   ├── train.py           ← V3 训练脚本
#   │   ├── densenet3d.py      ← DenseNet3D backbone 实现
#   │   ├── mrrate_dataset.py  ← MR-RATE 数据集加载器
#   │   ├── run.sh             ← 训练启动脚本
#   │   └── watchdog.sh        ← 自动守护进程
#   ├── checkpoints/           ← 中间 checkpoint（如有）
#   └── logs/                  ← 训练日志
#       ├── v1_full_train.log
#       ├── v1_e9.log
#       └── v1_e10.log

DIRS = {
    f"{NAS_ROOT}/weights":     [f"{SRC}/outputs/report_gen/best_model.pt",
                                 f"{SRC}/outputs/report_gen/last_model.pt"],
    f"{NAS_ROOT}/code":        [f"{SRC}/src/encoder.py",
                                 f"{SRC}/src/densenet3d.py",
                                 f"{SRC}/server_code/mrrate_dataset.py",
                                 f"{SRC}/scripts/train.py",
                                 f"{SRC}/run.sh",
                                 f"{SRC}/watchdog.sh"],
    f"{NAS_ROOT}/checkpoints": [],
    f"{NAS_ROOT}/logs":        [],
}


def run(cmd):
    print(f"  RUN: {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0 and "No such file" not in r.stderr:
        print(f"  WARN: {r.stderr[:200]}")
    return r.stdout.strip()


def human_size(path):
    try:
        sz = os.path.getsize(path)
        if sz > 1024 ** 3:
            return f"{sz / 1024 ** 3:.2f} GB"
        elif sz > 1024 ** 2:
            return f"{sz / 1024 ** 2:.1f} MB"
        else:
            return f"{sz / 1024:.1f} KB"
    except:
        return "N/A"


def collect_logs():
    log_sources = [
        f"{SRC}/logs",
        f"{SRC}/outputs/report_gen",
    ]
    for ld in log_sources:
        if not os.path.isdir(ld):
            continue
        for f in os.listdir(ld):
            if f.endswith(('.log', '.txt')):
                DIRS[f"{NAS_ROOT}/logs"].append(os.path.join(ld, f))


def collect_checkpoints():
    ckpt_dir = f"{SRC}/checkpoints"
    if os.path.isdir(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            DIRS[f"{NAS_ROOT}/checkpoints"].append(os.path.join(ckpt_dir, f))


def main():
    print("=" * 60)
    print(f"MR-RATE NAS Backup Tool")
    print(f"Target: {NAS_ROOT}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if not os.path.isdir(NAS_ROOT):
        print(f"ERROR: NAS not mounted at {NAS_ROOT}")
        print("Please mount NAS first:")
        print("  mount -t cifs //10.154.32.108/shared /mnt/nas1/disk07 -o username=guest")
        sys.exit(1)

    run(f"mkdir -p {NAS_ROOT}/weights {NAS_ROOT}/code {NAS_ROOT}/checkpoints {NAS_ROOT}/logs")

    collect_logs()
    collect_checkpoints()

    total_files = 0
    total_size = 0

    for dst_dir, src_files in DIRS.items():
        for sf in src_files:
            if not os.path.exists(sf):
                print(f"  SKIP (not found): {sf}")
                continue
            bn = os.path.basename(sf)
            dst = os.path.join(dst_dir, bn)
            sz = os.path.getsize(sf)
            total_files += 1
            total_size += sz

            if os.path.exists(dst):
                dst_sz = os.path.getsize(dst)
                if dst_sz == sz:
                    print(f"  SKIP (same size): {bn}")
                    continue

            print(f"  COPY {bn} ({human_size(sf)}) ...")
            t0 = time.time()
            shutil.copy2(sf, dst)
            elapsed = time.time() - t0
            speed = sz / elapsed / 1024 ** 2 if elapsed > 0 else 0
            print(f"    done in {elapsed:.1f}s ({speed:.1f} MB/s)")

    if total_files:
        print(f"\nTotal: {total_files} files, {total_size / 1024 ** 3:.2f} GB")

    # Write README (or copy from code dir if already there)
    readme_src = f"{NAS_ROOT}/code/README.md"
    readme_dst = f"{NAS_ROOT}/README.md"
    if os.path.exists(readme_src) and not os.path.exists(readme_dst):
        shutil.copy2(readme_src, readme_dst)
        print("README.md copied to root")

    print("\nNAS backup complete!")
    run(f"ls -lhR {NAS_ROOT}")


if __name__ == "__main__":
    main()
