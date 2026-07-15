import paramiko, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("10.176.60.72", username="jiaqigu", password="lijia7272", timeout=15)

BASE = '/home/jiaqigu/mrrate_hidnet/outputs/pretrain_densenet'
MODS = ['t1', 't2', 'flair']
COLORS = {'t1': '#4f6ef6', 't2': '#ef4444', 'flair': '#10b981'}
TITLES = {'t1': 'T1', 't2': 'T2', 'flair': 'FLAIR'}

all_data = {}

for mod in MODS:
    path = f"{BASE}/log_{mod}.jsonl"
    stdin, stdout, stderr = ssh.exec_command(f"cat {path}", get_pty=True)
    raw = stdout.read().decode(errors='ignore').strip()

    epochs, losses = [], []
    for line in raw.split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            ep = d.get('epoch', 0)
            lo = d.get('train_loss', 0)
            if ep > 0:
                epochs.append(ep)
                losses.append(lo)
        except:
            pass

    if epochs:
        # 确保按 epoch 排序
        paired = sorted(zip(epochs, losses))
        epochs, losses = zip(*paired)
        all_data[mod] = (list(epochs), list(losses))
        print(f"{mod}: {len(epochs)} epochs, loss {min(losses):.4f} -> {losses[-1]:.4f}")

ssh.close()

if not all_data:
    print("No data found!")
    exit(1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (mod, (eps, los)) in zip(axes, all_data.items()):
    color = COLORS[mod]
    ax.plot(eps, los, color=color, linewidth=1, alpha=0.7)
    ax.plot(eps, los, 'o', color=color, markersize=3, alpha=0.4)

    # SMA
    w = 5
    if len(los) >= w:
        sma = [sum(los[i:i+w])/w for i in range(len(los)-w+1)]
        ax.plot(eps[w-1:], sma, color='#333333', linewidth=2, alpha=0.6, label=f'{w}-epoch SMA')

    ax.set_title(TITLES[mod], fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_ylim(0, max(los) * 1.08)
    ax.grid(True, alpha=0.2)

    best = min(los)
    idx = los.index(best)
    ax.scatter([eps[idx]], [best], marker='*', s=200, color='#ff6b35', zorder=10,
               edgecolors='white', linewidth=1)
    ax.text(0.98, 0.95, f'Best: {best:.4f} (epoch {eps[idx]})\nEpochs: {len(los)}',
            transform=ax.transAxes, fontsize=9, va='top', ha='right', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#fef3c7', alpha=0.7, edgecolor='#d4a017'))

fig.suptitle('Pretrain DenseNet — Single-Modality Loss Curves', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()

out_dir = os.path.dirname(__file__)
for mod in MODS:
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    if mod in all_data:
        eps, los = all_data[mod]
        color = COLORS[mod]
        ax2.plot(eps, los, color=color, linewidth=1, alpha=0.7)

        w = 5
        if len(los) >= w:
            sma = [sum(los[i:i+w])/w for i in range(len(los)-w+1)]
            ax2.plot(eps[w-1:], sma, color='#333333', linewidth=2.5, alpha=0.7, label=f'{w}-epoch SMA')

        ax2.set_title(f'Pretrain Loss — {TITLES[mod]} ({len(los)} epochs)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_ylim(0, max(los) * 1.08)
        ax2.grid(True, alpha=0.2)

        best = min(los)
        bidx = los.index(best)
        ax2.scatter([eps[bidx]], [best], marker='*', s=250, color='#ff6b35', zorder=10,
                    edgecolors='white', linewidth=1)
        ax2.text(0.98, 0.95, f'Best loss: {best:.4f} (epoch {eps[bidx]})\nFinal loss: {los[-1]:.4f}',
                transform=ax2.transAxes, fontsize=10, va='top', ha='right', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#fef3c7', alpha=0.7, edgecolor='#d4a017'))
        ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'pretrain_{mod}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)

plt.savefig(os.path.join(out_dir, 'pretrain_combined.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"\nCombined: {out_dir}/pretrain_combined.png")
for mod in MODS:
    print(f"  {mod}: {out_dir}/pretrain_{mod}.png")
