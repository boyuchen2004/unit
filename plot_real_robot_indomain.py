"""
Real Robot In-Domain 实验结果可视化
配色：蓝黄色系，紧凑布局
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
    'xtick.labelsize': 11,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.unicode_minus': False,
})

# ============== 读取数据 ==============
DATA_PATH = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/data/real_robot_indomain.csv'
OUTPUT_DIR = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/'

df = pd.read_csv(DATA_PATH)

methods = df['Method'].tolist()
n_methods = len(methods)
_method_display = {
    'Factory Pretrain Only': 'VLA-UniT w/o human data',
    'Egodex+Factory': 'VLA-UniT w/ human data',
}
methods_display = [_method_display.get(m, m) for m in methods]

# 解析分数数据
def parse_fraction(frac_str):
    num, denom = frac_str.split('/')
    return int(num), int(denom)

# 类别列表
category_cols = ['Pick and Place', 'Bimanual Cooperation']

# 构建类别数据
categories = []
for col in category_cols:
    values = []
    n_trials_list = []
    for val in df[col]:
        num, denom = parse_fraction(val)
        values.append(num / denom)
        n_trials_list.append(denom)
    categories.append((col, np.array(values), n_trials_list[0]))

n_categories = len(categories)

# ============== 配色（Egodex+Factory 最佳 -> 黄色）==============
method_colors = [
    '#C5DCF2',   # GR00T-Qwen2.5 - 淡蓝
    '#7BA3D0',   # Factory Pretrain Only - 中蓝
    '#F5D76E',   # Egodex+Factory - 柔黄（最佳）
]

# ============== 计算标准误差 ==============
def calc_se(p, n):
    if p == 0 or p == 1:
        return 0.01
    return np.sqrt(p * (1 - p) / n)

# ============== 绘图 ==============
fig, ax = plt.subplots(figsize=(7, 5))

group_width = n_methods + 1.0
bar_width = 0.70

# 绘制每个类别
for cat_idx, (cat_name, values, n_samples) in enumerate(categories):
    se = np.array([calc_se(p, n_samples) for p in values])
    
    base_x = cat_idx * group_width
    x_positions = base_x + np.arange(n_methods)
    
    # 阴影
    shadow_offset_x = 0.08
    shadow_offset_y = -0.01
    shadow_values = np.maximum(values + shadow_offset_y, 0)
    ax.bar(x_positions + shadow_offset_x, shadow_values, bar_width, 
           color='#AAAAAA', edgecolor='none', zorder=1, alpha=0.55)
    
    # 主柱状图
    bars = ax.bar(x_positions, values, bar_width, yerr=se, 
                  color=method_colors,
                  edgecolor='#333333', linewidth=0.5, capsize=2,
                  error_kw={'elinewidth': 0.7, 'capthick': 0.7, 'ecolor': '#555555'},
                  zorder=2)
    
    # 数值标注（百分比）
    best = np.argmax(values)
    for i, (bar, val, err) in enumerate(zip(bars, values, se)):
        y_pos = val + err + 0.03
        label = f'{val*100:.0f}'
        
        if i == best:
            ax.annotate(label, xy=(x_positions[i], y_pos), ha='center', va='bottom',
                        fontsize=9.5, fontweight='bold', color='#1A1A1A')
            bar.set_edgecolor('#1A1A1A')
            bar.set_linewidth(1.0)
        else:
            ax.annotate(label, xy=(x_positions[i], y_pos), ha='center', va='bottom',
                        fontsize=9, color='#555555')

# X轴类别标签
category_centers = [cat_idx * group_width + (n_methods - 1) / 2 for cat_idx in range(n_categories)]
ax.set_xticks(category_centers)
ax.set_xticklabels([cat[0] for cat in categories], fontsize=13, fontweight='bold')

# 样本量标注
for cat_idx, (cat_name, values, n_samples) in enumerate(categories):
    base_x = cat_idx * group_width
    x_right = base_x + n_methods - 0.5
    ax.text(x_right, 0.92, f'n={n_samples}', ha='right', va='top',
            fontsize=9, color='#777777', style='italic')

# Y轴设置（百分比格式）
ax.set_ylim(0, 0.95)
ax.set_yticks(np.arange(0, 1.0, 0.2))
ax.set_yticklabels([f'{int(x*100)}%' for x in np.arange(0, 1.0, 0.2)])
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

# 浅蓝色背景（In-Domain）
ax.axvspan(-0.5, n_categories * group_width - 0.5, alpha=0.08, color='#5B8DC9', zorder=0)

ax.set_xlim(-0.5, n_categories * group_width - 0.5)

# 图例放在底部
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=method_colors[i], edgecolor='#444444', linewidth=0.4, label=methods_display[i]) 
                   for i in range(n_methods)]
fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02),
           ncol=3, frameon=False, handlelength=1.4, handletextpad=0.4, 
           columnspacing=0.8, fontsize=11)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)

plt.savefig(f'{OUTPUT_DIR}real_robot_indomain.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(f'{OUTPUT_DIR}real_robot_indomain.pdf', 
            bbox_inches='tight', facecolor='white', edgecolor='none')

print("✓ real_robot_indomain.png/pdf")
