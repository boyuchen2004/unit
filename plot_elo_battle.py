"""
ELO Battle 胜率对比图
- 水平堆叠条形图展示 Latent Action Win / Tie / Raw Action Win 比例
配色：蓝黄色系
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============== 论文级别配置 ==============
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.unicode_minus': False,
})

# ============== 读取数据 ==============
DATA_PATH = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/data/elo_battle.csv'
OUTPUT_DIR = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/'

df = pd.read_csv(DATA_PATH)

# ============== 数据准备 ==============
directions = df['Direction'].tolist()
latent_win = df['Latent_Win'].values
tie = df['Tie'].values
raw_win = df['Raw_Win'].values

# 配色
color_latent = '#F5D76E'   # 柔黄 - Latent Action Win
color_tie = '#E0E0E0'      # 灰色 - Tie
color_raw = '#5B8DC9'      # 柔蓝 - Raw Action Win

# ============== 绘图 ==============
fig, ax = plt.subplots(figsize=(10, 4))

y_pos = np.arange(len(directions))
bar_height = 0.5

# 水平堆叠条形图
# Latent Action Win (左侧，从0开始向右)
bars_latent = ax.barh(y_pos, latent_win, bar_height, 
                       color=color_latent, edgecolor='#333333', linewidth=0.8,
                       label='Latent Action Win')

# Tie (中间)
bars_tie = ax.barh(y_pos, tie, bar_height, left=latent_win,
                    color=color_tie, edgecolor='#333333', linewidth=0.8,
                    label='Tie')

# Raw Action Win (右侧)
bars_raw = ax.barh(y_pos, raw_win, bar_height, left=latent_win + tie,
                    color=color_raw, edgecolor='#333333', linewidth=0.8,
                    label='Raw Action Win')

# 数值标注
for i, (lw, t, rw) in enumerate(zip(latent_win, tie, raw_win)):
    # Latent Win 标注
    if lw > 0.08:
        ax.text(lw / 2, i, f'{lw*100:.1f}%', ha='center', va='center',
                fontsize=11, fontweight='bold', color='#333333')
    
    # Tie 标注
    if t > 0.08:
        ax.text(lw + t / 2, i, f'{t*100:.1f}%', ha='center', va='center',
                fontsize=10, color='#555555')
    
    # Raw Win 标注
    if rw > 0.08:
        ax.text(lw + t + rw / 2, i, f'{rw*100:.1f}%', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')

# Y轴标签
direction_labels = [d.replace('2', ' → ') for d in directions]
ax.set_yticks(y_pos)
ax.set_yticklabels(direction_labels, fontsize=12, fontweight='bold')

# X轴设置
ax.set_xlim(0, 1)
ax.set_xticks(np.arange(0, 1.1, 0.2))
ax.set_xticklabels([f'{int(x*100)}%' for x in np.arange(0, 1.1, 0.2)])
ax.set_xlabel('Win Rate', fontweight='bold')

# 50% 参考线
ax.axvline(x=0.5, color='#333333', linestyle='--', linewidth=1.2, alpha=0.7, zorder=0)

# 网格
ax.xaxis.grid(True, linestyle='--', alpha=0.3, color='#AAAAAA', linewidth=0.6)
ax.set_axisbelow(True)

# 边框
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_color('#555555')
ax.spines['bottom'].set_color('#555555')

# 图例
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          ncol=3, frameon=False, handlelength=1.5, handletextpad=0.5, 
          columnspacing=2.0, fontsize=10)

plt.title('ELO Battle: Latent Action vs Raw Action', fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)

plt.savefig(f'{OUTPUT_DIR}elo_battle.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{OUTPUT_DIR}elo_battle.pdf', bbox_inches='tight', facecolor='white')
print("✓ elo_battle.png/pdf")

# ============== 打印数据摘要 ==============
print("\n=== ELO Battle 胜率摘要 ===")
print(f"{'Direction':<15} {'Latent Win':>12} {'Tie':>10} {'Raw Win':>12}")
print("-" * 50)
for i, d in enumerate(directions):
    print(f"{d:<15} {latent_win[i]*100:>11.1f}% {tie[i]*100:>9.1f}% {raw_win[i]*100:>11.1f}%")
print("-" * 50)
avg_latent = latent_win.mean()
avg_tie = tie.mean()
avg_raw = raw_win.mean()
print(f"{'Average':<15} {avg_latent*100:>11.1f}% {avg_tie*100:>9.1f}% {avg_raw*100:>11.1f}%")
