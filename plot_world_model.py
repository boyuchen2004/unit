"""
World Model 实验结果可视化
- 雷达图：使用 PSNR 指标，三个数据集作为三个轴
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import pi

# ============== 论文级别配置 ==============
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 9,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.unicode_minus': False,
})

# ============== 读取数据 ==============
DATA_PATH = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/data/world_model.csv'
OUTPUT_DIR = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/'

df = pd.read_csv(DATA_PATH)

# ============== 数据准备 ==============
datasets = ['Droid', 'Egodex', 'Robocasa']
n_datasets = len(datasets)

# 提取 PSNR 值
raw_psnr = []
latent_psnr = []
for dataset in datasets:
    raw_row = df[(df['Dataset'] == dataset) & (df['Method'] == 'Raw Action')].iloc[0]
    latent_row = df[(df['Dataset'] == dataset) & (df['Method'] == 'Latent Action')].iloc[0]
    raw_psnr.append(raw_row['PSNR'])
    latent_psnr.append(latent_row['PSNR'])

raw_psnr = np.array(raw_psnr)
latent_psnr = np.array(latent_psnr)

# ============== 雷达图配置 ==============
# 角度设置（三个轴，均匀分布）
angles = [n / float(n_datasets) * 2 * pi for n in range(n_datasets)]
angles += angles[:1]  # 闭合

# 闭合数据
raw_values = list(raw_psnr) + [raw_psnr[0]]
latent_values = list(latent_psnr) + [latent_psnr[0]]

# 配色（莫兰迪色）
color_raw = '#5B8DC9'      # 柔蓝
color_latent = '#F5D76E'   # 柔黄

# ============== 绘制雷达图 ==============
fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(projection='polar'))

# 设置起始角度（从顶部开始）
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# 绘制 Raw Action
ax.plot(angles, raw_values, 'o-', linewidth=2.5, color=color_raw, 
        label='Raw Action', markersize=8, markeredgecolor='white', markeredgewidth=1)
ax.fill(angles, raw_values, alpha=0.25, color=color_raw)

# 绘制 Latent Action
ax.plot(angles, latent_values, 's-', linewidth=2.5, color=color_latent, 
        label='Latent Action', markersize=8, markeredgecolor='white', markeredgewidth=1)
ax.fill(angles, latent_values, alpha=0.35, color=color_latent)

# 设置轴标签（数据集名称）
ax.set_xticks(angles[:-1])
ax.set_xticklabels(datasets, fontsize=12, fontweight='bold')

# Y轴范围和刻度（PSNR 范围）
y_min = 10
y_max = 30
ax.set_ylim(y_min, y_max)
y_ticks = [12, 16, 20, 24, 28]
ax.set_yticks(y_ticks)
ax.set_yticklabels([f'{y}' for y in y_ticks], fontsize=9, color='#666666')

# 网格样式
ax.grid(True, linestyle='--', alpha=0.5, color='#AAAAAA', linewidth=0.6)

# 在每个数据点旁边标注数值
for i, (dataset, raw_v, latent_v) in enumerate(zip(datasets, raw_psnr, latent_psnr)):
    angle = angles[i]
    
    # 计算标注位置偏移
    # Raw Action 标注（内侧）
    raw_offset = -1.8
    ax.annotate(f'{raw_v:.2f}', 
                xy=(angle, raw_v), 
                xytext=(angle, raw_v + raw_offset),
                ha='center', va='center',
                fontsize=9, color=color_raw, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color_raw, alpha=0.8, linewidth=0.5))
    
    # Latent Action 标注（外侧）
    latent_offset = 2.2
    ax.annotate(f'{latent_v:.2f}', 
                xy=(angle, latent_v), 
                xytext=(angle, latent_v + latent_offset),
                ha='center', va='center',
                fontsize=9, color='#5A4A3A', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color_latent, alpha=0.8, linewidth=0.5))

# 图例
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), frameon=True, 
          fancybox=True, shadow=False, edgecolor='#CCCCCC', fontsize=10)

# 标题
plt.title('PSNR Comparison (dB) ↑', fontsize=14, fontweight='bold', pad=25)

plt.tight_layout()

plt.savefig(f'{OUTPUT_DIR}world_model_radar.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{OUTPUT_DIR}world_model_radar.pdf', bbox_inches='tight', facecolor='white')
print("✓ world_model_radar.png/pdf")

# ============== 打印数据摘要 ==============
print("\n=== PSNR 数据摘要 ===")
print(f"{'Dataset':<12} {'Raw Action':>12} {'Latent Action':>14} {'Improvement':>12}")
print("-" * 52)
for i, dataset in enumerate(datasets):
    imp = (latent_psnr[i] - raw_psnr[i]) / raw_psnr[i] * 100
    print(f"{dataset:<12} {raw_psnr[i]:>12.2f} {latent_psnr[i]:>14.2f} {imp:>11.1f}%")
