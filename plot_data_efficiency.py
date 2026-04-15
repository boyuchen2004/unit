"""
Data Efficiency 可视化
比较 Groot-baseline 和 Unified-token 在不同数据规模下的 Overall 成功率
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
    'xtick.labelsize': 12,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.unicode_minus': False,
})

# ============== 读取数据 ==============
DATA_PATH = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/data/data_efficiency.csv'
OUTPUT_DIR = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/'

df = pd.read_csv(DATA_PATH)

# 数据整理（Method 列与 CSV 一致：GR00T-Qwen2.5 / VLA-UniT）
groot_100 = df[(df['Method'] == 'GR00T-Qwen2.5') & (df['Data Scale'] == '24×100 trajs')]['Overall'].values[0]
unified_100 = df[(df['Method'] == 'VLA-UniT') & (df['Data Scale'] == '24×100 trajs')]['Overall'].values[0]

groot_1000 = df[(df['Method'] == 'GR00T-Qwen2.5') & (df['Data Scale'] == '24×1000 trajs')]['Overall'].values[0]
unified_1000 = df[(df['Method'] == 'VLA-UniT') & (df['Data Scale'] == '24×1000 trajs')]['Overall'].values[0]

# ============== 蓝黄配色 ==============
color_groot = '#5B8DC9'      # 柔蓝 (GR00T-Qwen2.5)
color_unified = '#F5D76E'    # 柔黄 (VLA-UniT)
method_colors = [color_groot, color_unified]
methods = ['GR00T-Qwen2.5', 'VLA-UniT']

# ============== 计算标准误差 ==============
n_total = 24 * 50  # 1200 trials
def calc_se(p):
    return np.sqrt(p * (1 - p) / n_total)

se_groot_100 = calc_se(groot_100)
se_unified_100 = calc_se(unified_100)
se_groot_1000 = calc_se(groot_1000)
se_unified_1000 = calc_se(unified_1000)

# ============== 绘图 ==============
fig, ax = plt.subplots(figsize=(6, 5))

# 类别数据
categories = [
    ('24×100 trajs', [groot_100, unified_100], [se_groot_100, se_unified_100], n_total),
    ('24×1000 trajs', [groot_1000, unified_1000], [se_groot_1000, se_unified_1000], n_total),
]

n_methods = 2
n_categories = len(categories)
group_width = n_methods + 1.0
bar_width = 0.70

# 绘制每个类别
for cat_idx, (cat_name, values, se_vals, n_samples) in enumerate(categories):
    base_x = cat_idx * group_width
    x_positions = base_x + np.arange(n_methods)
    
    # 阴影（向右下偏移）
    shadow_offset_x = 0.10
    shadow_offset_y = -0.012
    shadow_values = np.maximum(np.array(values) - 0.10 + shadow_offset_y, 0)
    ax.bar(x_positions + shadow_offset_x, shadow_values, bar_width, 
           color='#AAAAAA', edgecolor='none', zorder=1, alpha=0.55,
           bottom=0.10)
    
    # 主柱状图
    bars = ax.bar(x_positions, np.array(values) - 0.10, bar_width, yerr=se_vals, 
                  color=method_colors,
                  edgecolor='#333333', linewidth=0.5, capsize=2,
                  error_kw={'elinewidth': 0.7, 'capthick': 0.7, 'ecolor': '#555555'},
                  zorder=2, bottom=0.10)
    
    # 数值标注
    best = np.argmax(values)
    for i, (bar, val, err) in enumerate(zip(bars, values, se_vals)):
        y_pos = val + err + 0.012
        label = f'{val*100:.1f}'
        
        if i == best:
            ax.annotate(label, xy=(x_positions[i], y_pos), ha='center', va='bottom',
                        fontsize=9.5, fontweight='bold', color='#1A1A1A')
            bar.set_edgecolor('#1A1A1A')
            bar.set_linewidth(1.0)
        else:
            ax.annotate(label, xy=(x_positions[i], y_pos), ha='center', va='bottom',
                        fontsize=7.5, color='#555555')

# X轴类别标签
category_centers = [cat_idx * group_width + (n_methods - 1) / 2 for cat_idx in range(n_categories)]
ax.set_xticks(category_centers)
ax.set_xticklabels([cat[0] for cat in categories], fontsize=12, fontweight='bold')

# 样本量标注（在每组柱子右上角）
for cat_idx, (cat_name, values, se_vals, n_samples) in enumerate(categories):
    base_x = cat_idx * group_width
    x_right = base_x + n_methods - 0.5
    ax.text(x_right, 0.76, f'n={n_samples}', ha='right', va='top',
            fontsize=7, color='#777777', style='italic')

# Y轴设置（百分比格式）
ax.set_ylim(0.10, 0.78)
ax.set_yticks(np.arange(0.1, 0.80, 0.1))
ax.set_yticklabels([f'{int(x*100)}%' for x in np.arange(0.1, 0.80, 0.1)])
ax.set_ylabel('Success Rate', fontweight='bold')

# 网格（虚线）
ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#AAAAAA', linewidth=0.6)
ax.set_axisbelow(True)

# 边框
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_color('#555555')
ax.spines['left'].set_linewidth(0.6)
ax.spines['bottom'].set_color('#555555')
ax.spines['bottom'].set_linewidth(0.6)
ax.tick_params(bottom=False, left=True, colors='#333333', length=3, width=0.6)

# 整体浅蓝色背景
ax.axvspan(-0.8, n_categories * group_width - 1 + 0.5, alpha=0.08, color='#5B8DC9', zorder=0)

ax.set_xlim(-0.8, n_categories * group_width - 1 + 0.5)

# 图例放在底部
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, edgecolor='#444444', linewidth=0.4, label=m) 
                   for m, c in zip(methods, method_colors)]
fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02),
           ncol=2, frameon=False, handlelength=1.4, handletextpad=0.4, 
           columnspacing=1.0, fontsize=11)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

plt.savefig(f'{OUTPUT_DIR}data_efficiency.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(f'{OUTPUT_DIR}data_efficiency.pdf', 
            bbox_inches='tight', facecolor='white', edgecolor='none')

print("✓ data_efficiency.png/pdf")
