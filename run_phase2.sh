#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
source ~/hidnet_env/bin/activate
cd /home/jiaqigu/mrrate_hidnet
rm -f outputs/report_gen/phase2.log outputs/report_gen/phase2_latest.pt outputs/report_gen/phase2_step*.pt
python train_v5_phase2.py \
  --encoder_ckpt /home/jiaqigu/mrrate_hidnet/outputs/report_gen/phase1_latest.pt \
  --qwen_path /mnt/nas1/disk07/public/model_weights/Qwen2.5-3B-Instruct \
  --data_root /mnt/nas1/disk07/public/mr_data/MR-RATE \
  --log_dir /home/jiaqigu/mrrate_hidnet/outputs/report_gen \
  --batch_id batch27 \
  --epochs 3 \
  --batch_size 4 \
  --lr 1e-4 \
  --max_text_len 256 \
  2>&1 | tee outputs/report_gen/phase2.log
