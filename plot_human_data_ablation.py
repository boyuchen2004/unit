"""
Human Data Ablation 可视化
比较有无 human data 在不同数据规模下的表现
两组对比：100 trajs vs 100 trajs w/ human, 1000 trajs vs 1000 trajs w/ human
配色：蓝黄色系，100 trajs 组凸显
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
DATA_PATH = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/data/human_data_ablation.csv'
OUTPUT_DIR = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/'

df = pd.read_csv(DATA_PATH)

# 只保留 100 trajs 的两组
df_100 = df.iloc[:2].reset_index(drop=True)
methods = df_100['Method'].tolist()
n_methods = len(methods)

# In-Domain 数据 (只用 100 trajs 的两组)
in_domain_avg = df_100['In-Domain Avg'].values
n_in_domain = 24 * 50

# OOD 数据 (只用 100 trajs 的两组)
ood_appearance = df_100['Unseen Obj Appearance'].values  # 18×50 = 900
ood_container = df_100['Unseen Obj-Container Combs'].values
ood_obj_type = df_100['OOD Avg'].values

n_ood_appearance = 18 * 50
n_ood_container = 23 * 50
n_ood_type = 32 * 50
n_ood_total = n_ood_appearance + n_ood_container + n_ood_type

# 计算 OOD Average (3类加权平均)
ood_avg = (ood_appearance * n_ood_appearance + ood_container * n_ood_container + ood_obj_type * n_ood_type) / n_ood_total

# 所有类别数据 (7个类别，详细展开)
pick_place = df_100['Pick and Place'].values
articulated = df_100['Articulated'].values
n_pp = 18 * 50
n_art = 6 * 50

# X 轴显示名：仅 Unseen* 在 Unseen 后换行（不改图幅/组距）
categories = [
    ('Pick and Place', pick_place, n_pp),
    ('Articulated', articulated, n_art),
    ('In-Domain Average', in_domain_avg, n_in_domain),
    ('Unseen\nAppearance', ood_appearance, n_ood_appearance),
    ('Unseen\nCombinations', ood_container, n_ood_container),
    ('Unseen\nObject Types', ood_obj_type, n_ood_type),
    ('OOD Average', ood_avg, n_ood_total),
]

n_categories = len(categories)

# ============== 配色（100 组）==============
# 蓝色系 = 无 human data，黄色系 = 有 human data
method_colors = [
    '#5B8DC9',   # 100 GR1 trajs - 柔蓝
    '#F5D76E',   # 100 GR1 trajs w/ human - 柔黄（highlight）
]

# 图例单行；用更宽画布 + columnspacing 拉开，不分行
methods_short = [
    'VLA-UniT w/o human data',
    'VLA-UniT w/ human data',
]

# ============== 计算标准误差 ==============
def calc_se(p, n):
    return np.sqrt(p * (1 - p) / n)

# ============== 绘图 ==============
fig, ax = plt.subplots(figsize=(12, 5.5))

# 每个类别的位置
group_width = n_methods + 1.5
bar_width = 0.70

# 绘制每个类别
for cat_idx, (cat_name, values, n_samples) in enumerate(categories):
    se = np.array([calc_se(p, n_samples) for p in values])
    
    base_x = cat_idx * group_width
    x_positions = base_x + np.arange(n_methods)
    
    # 阴影
    shadow_offset_x = 0.10
    shadow_offset_y = -0.012
    shadow_values = np.maximum(values - 0.20 + shadow_offset_y, 0)
    ax.bar(x_positions + shadow_offset_x, shadow_values, bar_width, 
           color='#AAAAAA', edgecolor='none', zorder=1, alpha=0.55,
           bottom=0.20)
    
    # 主柱状图
    bars = ax.bar(x_positions, values - 0.20, bar_width, yerr=se, 
                  color=method_colors,
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
                        fontsize=9, color='#555555')

# X轴类别标签
category_centers = [cat_idx * group_width + (n_methods - 1) / 2 for cat_idx in range(n_categories)]
ax.set_xticks(category_centers)
ax.set_xticklabels([cat[0] for cat in categories], fontsize=11, fontweight='bold', rotation=0)

# 样本量标注
for cat_idx, (cat_name, values, n_samples) in enumerate(categories):
    base_x = cat_idx * group_width
    x_right = base_x + n_methods - 0.5
    ax.text(x_right, 0.68, f'n={n_samples}', ha='right', va='top',
            fontsize=9, color='#777777', style='italic')

# Y轴设置
# Y轴设置（百分比格式）
ax.set_ylim(0.20, 0.70)
ax.set_yticks(np.arange(0.2, 0.75, 0.1))
ax.set_yticklabels([f'{int(x*100)}%' for x in np.arange(0.2, 0.75, 0.1)])
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

# 分区标注
# In-Domain 区域 (前3个类别)
in_domain_start = -0.8
in_domain_end = 3 * group_width - 1.5
ax.axvspan(in_domain_start, in_domain_end, alpha=0.08, color='#5B8DC9', zorder=0)
ax.text((in_domain_start + in_domain_end) / 2, 0.68, 'In-Domain', 
        ha='center', va='bottom', fontsize=13, fontweight='bold', color='#3A6EA5')

# OOD 区域 (后4个类别)
ood_start = in_domain_end + 0.3
ood_end = n_categories * group_width - 1
ax.axvspan(ood_start, ood_end, alpha=0.08, color='#F5D76E', zorder=0)
ax.text((ood_start + ood_end) / 2, 0.68, 'OOD', 
        ha='center', va='bottom', fontsize=13, fontweight='bold', color='#B8860B')

ax.set_xlim(in_domain_start, ood_end + 0.5)

# 图例放在底部
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=method_colors[i], edgecolor='#444444', linewidth=0.4, label=methods_short[i]) 
                   for i in range(n_methods)]
fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02),
           ncol=2, frameon=False, handlelength=1.4, handletextpad=0.4,
           columnspacing=1.0, fontsize=10)

plt.tight_layout()
plt.subplots_adjust(bottom=0.16)

plt.savefig(f'{OUTPUT_DIR}human_data_ablation.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(f'{OUTPUT_DIR}human_data_ablation.pdf', 
            bbox_inches='tight', facecolor='white', edgecolor='none')

print("✓ human_data_ablation.png/pdf")
