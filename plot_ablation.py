"""
Ablation Study 可视化
一张图展示所有类别，分区标注 In-Domain 和 OOD
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
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.unicode_minus': False,
})

# ============== 读取数据 ==============
DATA_PATH = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/data/ablation.csv'
OUTPUT_DIR = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/'

df = pd.read_csv(DATA_PATH)

# 方法顺序倒序，Unified-token 放最右边
methods = df['Method'].tolist()[::-1]  # 倒序
n_methods = len(methods)

# 数据也需要倒序
df_reversed = df.iloc[::-1].reset_index(drop=True)

# In-Domain 数据 (使用倒序后的数据)
in_domain_avg = df_reversed['In-Domain Avg'].values
n_in_domain = 24 * 50  # 1200

# OOD 数据 (使用倒序后的数据)
ood_appearance = df_reversed['Unseen Obj Appearance'].values  # 18×50 = 900
ood_container = df_reversed['Unseen Obj-Container Combs'].values  # 32×50 = 1600
# Unseen Obj-Type 是 OOD Pick and Place Avg，已经是 articulated + pick place 的均值
ood_obj_type = df_reversed['OOD Pick and Place Avg'].values  # 23×50 = 1150

n_ood_appearance = 18 * 50
n_ood_container = 32 * 50
n_ood_type = 23 * 50
n_ood_total = n_ood_appearance + n_ood_container + n_ood_type

# 计算 OOD Average (3类加权平均)
ood_avg = (ood_appearance * n_ood_appearance + ood_container * n_ood_container + ood_obj_type * n_ood_type) / n_ood_total

# 所有类别数据 (7个类别，使用倒序后的数据)
categories = [
    ('Pick and Place', df_reversed['Pick and Place'].values, 18 * 50),
    ('Articulated', df_reversed['Articulated'].values, 6 * 50),
    ('Average', in_domain_avg, n_in_domain),
    ('Unseen Obj\nAppearance', ood_appearance, n_ood_appearance),
    ('Unseen Obj-Container\nCombinations', ood_container, n_ood_container),
    ('Unseen Obj-Type', ood_obj_type, n_ood_type),
    ('Average', ood_avg, n_ood_total),
]

n_categories = len(categories)

# ============== 蓝黄渐变配色（4个方法，倒序后）==============
# 顺序: Vision-token, Action-token, w/o cross reconstruction, Unified-token
method_colors = [
    '#5B8DC9',   # Vision-token - 柔蓝
    '#7BA3D0',   # Action-token - 中蓝
    '#B8D4F0',   # w/o cross reconstruction - 淡蓝
    '#F5D76E',   # Unified-token - 柔黄（最佳，最右边）
]

# ============== 计算标准误差 ==============
def calc_se(p, n):
    return np.sqrt(p * (1 - p) / n)

# ============== 绘图 ==============
fig, ax = plt.subplots(figsize=(15, 5.5))

# 每个类别的位置
group_width = n_methods + 1.5  # 每组宽度（含间隔）
bar_width = 0.75

# 绘制每个类别
for cat_idx, (cat_name, values, n_samples) in enumerate(categories):
    se = np.array([calc_se(p, n_samples) for p in values])
    
    # 该类别的基准x位置
    base_x = cat_idx * group_width
    x_positions = base_x + np.arange(n_methods)
    
    # 阴影（向右下偏移，更明显）
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
                        fontsize=7.5, fontweight='bold', color='#1A1A1A')
            bar.set_edgecolor('#1A1A1A')
            bar.set_linewidth(1.0)
        else:
            ax.annotate(label, xy=(x_positions[i], y_pos), ha='center', va='bottom',
                        fontsize=7, color='#555555')
    

# X轴类别标签（加粗）
category_centers = [cat_idx * group_width + (n_methods - 1) / 2 for cat_idx in range(n_categories)]
ax.set_xticks(category_centers)
ax.set_xticklabels([cat[0] for cat in categories], fontsize=10, fontweight='bold')

# 样本量标注（在每组柱子右上角）
for cat_idx, (cat_name, values, n_samples) in enumerate(categories):
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

# 分区标注
# In-Domain 区域 (前3个类别)
in_domain_start = -0.8
in_domain_end = 3 * group_width - 1.5
ax.axvspan(in_domain_start, in_domain_end, alpha=0.08, color='#5B8DC9', zorder=0)
ax.text((in_domain_start + in_domain_end) / 2, 0.76, 'In-Domain', 
        ha='center', va='bottom', fontsize=11, fontweight='bold', color='#3A6EA5')

# OOD 区域 (第4-7个类别)
ood_start = in_domain_end + 0.3
ood_end = n_categories * group_width - 1
ax.axvspan(ood_start, ood_end, alpha=0.08, color='#F5D76E', zorder=0)
ax.text((ood_start + ood_end) / 2, 0.76, 'OOD', 
        ha='center', va='bottom', fontsize=11, fontweight='bold', color='#B8860B')

ax.set_xlim(in_domain_start, ood_end + 0.5)

# 图例放在底部
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=method_colors[i], edgecolor='#444444', linewidth=0.4, label=methods[i]) 
                   for i in range(n_methods)]
ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.18),
          ncol=4, frameon=False, handlelength=1.4, handletextpad=0.4, 
          columnspacing=1.0, fontsize=10)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)

plt.savefig(f'{OUTPUT_DIR}ablation.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(f'{OUTPUT_DIR}ablation.pdf', 
            bbox_inches='tight', facecolor='white', edgecolor='none')

print("✓ ablation.png/pdf")
