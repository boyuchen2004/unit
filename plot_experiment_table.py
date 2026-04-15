"""
实验结果可视化 - 论文级别图表
配色方案：蓝黄渐变色系
数据源：CSV文件
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib

# ============== 论文级别配置 ==============
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.unicode_minus': False,
})

# ============== 读取CSV数据 ==============
DATA_PATH = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/data/experiment_results.csv'
OUTPUT_DIR = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/'

df = pd.read_csv(DATA_PATH)
methods_full = df['Method'].tolist()
articulated = df['Articulated'].values
pick_place = df['Pick and Place'].values
overall = df['Overall'].values

n_methods = len(methods_full)

# ============== 蓝黄渐变配色（低饱和高亮度）==============
blue_start = '#5B8DC9'   # 柔蓝
blue_end = '#C5DCF2'     # 淡蓝
yellow_start = '#FFF3B8' # 奶黄
yellow_end = '#F5D76E'   # 柔黄

def generate_gradient_colors(n, start_color, end_color):
    """生成渐变色"""
    cmap = LinearSegmentedColormap.from_list('custom', [start_color, end_color], N=n)
    return [matplotlib.colors.rgb2hex(cmap(i / (n - 1))) for i in range(n)]

# 前6个方法用蓝色渐变，后3个用黄色渐变
n_blue = 6
n_yellow = 3
blue_colors = generate_gradient_colors(n_blue, blue_start, blue_end)
yellow_colors = generate_gradient_colors(n_yellow, yellow_start, yellow_end)

method_colors = blue_colors + yellow_colors

# ============== 计算标准误差 ==============
n_pp, n_art, n_all = 18 * 50, 6 * 50, 24 * 50
se_art = np.sqrt(articulated * (1 - articulated) / n_art)
se_pp = np.sqrt(pick_place * (1 - pick_place) / n_pp)
se_all = np.sqrt(overall * (1 - overall) / n_all)

# ============== 绘图 ==============
fig, ax = plt.subplots(figsize=(15, 5.5))

# 每个类别的位置
group_width = n_methods + 1.5
bar_width = 0.75

# 类别数据
categories = [
    ('Pick and Place', pick_place, se_pp, n_pp),
    ('Articulated', articulated, se_art, n_art),
    ('Overall', overall, se_all, n_all),
]

n_categories = len(categories)

# 绘制每个类别
for cat_idx, (cat_name, values, se, n_samples) in enumerate(categories):
    base_x = cat_idx * group_width
    x_positions = base_x + np.arange(n_methods)
    
    # 阴影（向右下偏移）
    shadow_offset_x = 0.10
    shadow_offset_y = -0.012
    shadow_values = np.maximum(values - 0.20 + shadow_offset_y, 0)
    ax.bar(x_positions + shadow_offset_x, shadow_values, bar_width, 
           color='#AAAAAA', edgecolor='none', zorder=1, alpha=0.55,
           bottom=0.20)
    
    # 主柱状图
    bars = ax.bar(x_positions, values - 0.20, bar_width, yerr=se, color=method_colors,
                  edgecolor='#333333', linewidth=0.5, capsize=2,
                  error_kw={'elinewidth': 0.7, 'capthick': 0.7, 'ecolor': '#555555'},
                  zorder=2, bottom=0.20)
    
    # 数值标注
    best = np.argmax(values)
    for i, (bar, val, err) in enumerate(zip(bars, values, se)):
        y_pos = val + err + 0.010
        label = f'{val*100:.1f}'
        
        if i == best:
            ax.annotate(label, xy=(x_positions[i], y_pos), ha='center', va='bottom',
                        fontsize=9.5, fontweight='bold', color='#1A1A1A')
            bar.set_edgecolor('#1A1A1A')
            bar.set_linewidth(1.0)
        else:
            ax.annotate(label, xy=(x_positions[i], y_pos), ha='center', va='bottom',
                        fontsize=7, color='#555555')

# X轴类别标签（加粗）
category_centers = [cat_idx * group_width + (n_methods - 1) / 2 for cat_idx in range(n_categories)]
ax.set_xticks(category_centers)
ax.set_xticklabels([cat[0] for cat in categories], fontsize=12, fontweight='bold')

# 样本量标注（在每组柱子右上角）
for cat_idx, (cat_name, values, se, n_samples) in enumerate(categories):
    base_x = cat_idx * group_width
    x_right = base_x + n_methods - 0.5
    ax.text(x_right, 0.76, f'n={n_samples}', ha='right', va='top',
            fontsize=7, color='#777777', style='italic')

# Y轴设置（百分比格式）
ax.set_ylim(0.20, 0.78)
ax.set_yticks(np.arange(0.2, 0.80, 0.1))
ax.set_yticklabels([f'{int(x*100)}%' for x in np.arange(0.2, 0.80, 0.1)])
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
                   for m, c in zip(methods_full, method_colors)]
fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02),
           ncol=5, frameon=False, handlelength=1.4, handletextpad=0.4, 
           columnspacing=0.7, fontsize=11)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)

plt.savefig(f'{OUTPUT_DIR}method_comparison.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(f'{OUTPUT_DIR}method_comparison.pdf', 
            bbox_inches='tight', facecolor='white', edgecolor='none')

print("✓ method_comparison.png/pdf")
