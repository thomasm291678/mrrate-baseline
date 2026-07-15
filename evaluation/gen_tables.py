import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for f in fm.fontManager.ttflist:
    if "SimHei" in f.name:
        plt.rcParams["font.family"] = "SimHei"
        break
plt.rcParams["axes.unicode_minus"] = False

# ========== Table 1: Consistency ==========
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.axis("off")

col_labels = ["指标", "VAL（183样本）", "TEST（54样本）"]
rows = [
    ["Macro F1（参考）", "0.3835", "0.2463"],
    ["Micro F1（参考）", "0.6465", "0.6750"],
    ["Cohen's Kappa", "0.7743", "0.8801"],
    ["Macro F1（生成）", "0.4466", "0.2350"],
    ["Micro F1（生成）", "0.6776", "0.6806"],
]

table = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center",
                 colWidths=[0.30, 0.28, 0.28])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.15, 2.0)

for key, cell in table.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if key[0] == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("#f8f9fa" if key[0] % 2 == 1 else "#ffffff")

plt.title("DeepSeek V4 Pro vs Keyword —— 一致性指标", fontsize=15, fontweight="bold", pad=22, color="#2c3e50")
plt.tight_layout()
plt.savefig("C:/Users/HP/Documents/5555/table_consistency.png", dpi=200, bbox_inches="tight")
print("table_consistency.png saved")

# ========== Table 2: Density ==========
fig2, ax2 = plt.subplots(figsize=(8, 2.6))
ax2.axis("off")

col_labels2 = ["数据集", "Keyword 阳性", "LLM 阳性", "LLM / Keyword"]
rows2 = [
    ["VAL", "235（3.5%）", "260（3.8%）", "1.11×"],
    ["TEST", "72（3.6%）", "88（4.4%）", "1.22×"],
]

table2 = ax2.table(cellText=rows2, colLabels=col_labels2, cellLoc="center", loc="center",
                   colWidths=[0.16, 0.24, 0.24, 0.22])
table2.auto_set_font_size(False)
table2.set_fontsize(12)
table2.scale(1.15, 2.0)

for key, cell in table2.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if key[0] == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("#f8f9fa" if key[0] % 2 == 1 else "#ffffff")

plt.title("DeepSeek V4 Pro vs Keyword —— 标签密度", fontsize=15, fontweight="bold", pad=22, color="#2c3e50")
plt.tight_layout()
plt.savefig("C:/Users/HP/Documents/5555/table_density.png", dpi=200, bbox_inches="tight")
print("table_density.png saved")
