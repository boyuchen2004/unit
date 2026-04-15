"""
Zero-Shot 实验结果可视化
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
    'xtick.labelsize': 10,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.unicode_minus': False,
})

# ============== 读取数据 ==============
DATA_PATH = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/data/zero_shot.csv'
OUTPUT_DIR = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/'

df = pd.read_csv(DATA_PATH)

methods = df['Method'].tolist()
n_methods = len(methods)

# 解析分数数据
def parse_fraction(frac_str):
    num, denom = frac_str.split('/')
    return int(num), int(denom)

values = []
n_trials = []
for val in df['Zero-Shot']:
    num, denom = parse_fraction(val)
    values.append(num / denom)
    n_trials.append(denom)

values = np.array(values)
n_samples = n_trials[0]

# ============== 配色 ==============
method_colors = [
    '#C5DCF2',   # GR00T-Qwen2.5 - 淡蓝
    '#7BA3D0',   # VLA-UniT w/o Cross-Recon - 中蓝
    '#5B8DC9',   # VLA-UniT w/o Human - 柔蓝
    '#F5D76E',   # VLA-UniT w/ Human - 柔黄（highlight）
]

# ============== 计算标准误差 ==============
def calc_se(p, n):
    if p == 0 or p == 1:
        return 0.01  # 避免 SE=0
    return np.sqrt(p * (1 - p) / n)

se = np.array([calc_se(p, n_samples) for p in values])

# ============== 绘图 ==============
fig, ax = plt.subplots(figsize=(5, 5))

x = np.arange(n_methods)
bar_width = 0.65

# 阴影
shadow_offset_x = 0.08
shadow_offset_y = -0.01
shadow_values = np.maximum(values + shadow_offset_y, 0)
ax.bar(x + shadow_offset_x, shadow_values, bar_width,
       color='#AAAAAA', edgecolor='none', zorder=1, alpha=0.55)

# 主柱状图（不带误差棒）
bars = ax.bar(x, values, bar_width,
              color=method_colors,
              edgecolor='#333333', linewidth=0.5,
              zorder=2)

# 数值标注
best = np.argmax(values)
for i, (bar, val) in enumerate(zip(bars, values)):
    y_pos = val + 0.03
    label = f'{val*100:.0f}'

    if i == best:
        ax.annotate(label, xy=(i, y_pos), ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='#1A1A1A')
        bar.set_edgecolor('#1A1A1A')
        bar.set_linewidth(1.2)
    else:
        ax.annotate(label, xy=(i, y_pos), ha='center', va='bottom',
                    fontsize=8, color='#555555')

# X轴
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=9, fontweight='bold', rotation=15, ha='right')

# Y轴设置（百分比格式）
ax.set_ylim(0, 0.80)
ax.set_yticks(np.arange(0, 0.85, 0.2))
ax.set_yticklabels([f'{int(x*100)}%' for x in np.arange(0, 0.85, 0.2)])
ax.set_ylabel('Success Rate', fontweight='bold')

# 样本量
ax.text(0.97, 0.97, f'n={n_samples}', transform=ax.transAxes,
        ha='right', va='top', fontsize=8, color='#777777', style='italic')

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

# 浅黄色背景（OOD）
ax.axvspan(-0.5, n_methods - 0.5, alpha=0.08, color='#F5D76E', zorder=0)

ax.set_xlim(-0.5, n_methods - 0.5)

plt.tight_layout()

plt.savefig(f'{OUTPUT_DIR}zero_shot.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(f'{OUTPUT_DIR}zero_shot.pdf',
            bbox_inches='tight', facecolor='white', edgecolor='none')

print("✓ zero_shot.png/pdf")
