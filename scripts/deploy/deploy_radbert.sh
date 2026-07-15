#!/bin/bash
set -e

SERVER="jiaqigu@10.176.60.72"
REMOTE_DIR="/home/jiaqigu/mrrate_hidnet"

echo "=== Uploading test_radbert.py ==="
scp -o StrictHostKeyChecking=no "C:/Users/HP/Documents/5555/test_radbert.py" "${SERVER}:${REMOTE_DIR}/"

echo "=== Launching RadBERT test in tmux ==="
ssh -o StrictHostKeyChecking=no "$SERVER" << 'ENDSSH'
cd /home/jiaqigu/mrrate_hidnet
tmux kill-session -t radbert 2>/dev/null || true
tmux new-session -d -s radbert "source ~/hidnet_env/bin/activate && python test_radbert.py 2>&1 | tee /home/jiaqigu/mrrate_hidnet/radbert_result.log"
echo "RadBERT test started in tmux session 'radbert'"
ENDSSH

echo "=== Done ==="
echo "Check with: ssh jiaqigu@10.176.60.72 tmux attach -t radbert"
