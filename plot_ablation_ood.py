"""
Ablation Study - OOD Only
配色：蓝黄色系，统一命名
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

DATA_PATH = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/data/ablation.csv'
OUTPUT_DIR = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/'

df = pd.read_csv(DATA_PATH)

label_map = {
    'Unified-token': 'VLA-UniT',
    'w/o cross reconstruction': 'VLA-UniT w/o Cross-Recon',
    'Action-token': 'VLA-Action',
    'Vision-token': 'VLA-Vision',
}

display_order = ['Action-token', 'Vision-token', 'w/o cross reconstruction', 'Unified-token']
df_ordered = df.set_index('Method').loc[display_order].reset_index()
methods_display = [label_map[m] for m in display_order]
n_methods = len(methods_display)

df_reversed = df_ordered

ood_appearance = df_reversed['Unseen Obj Appearance'].values
ood_combinations = df_reversed['OOD Pick and Place Avg'].values
ood_obj_type = df_reversed['Unseen Obj-Container Combs'].values

n_ood_appearance = 18 * 50
n_ood_combinations = 23 * 50
n_ood_type = 32 * 50
n_ood_total = n_ood_appearance + n_ood_combinations + n_ood_type

ood_avg = (ood_appearance * n_ood_appearance + ood_combinations * n_ood_combinations + ood_obj_type * n_ood_type) / n_ood_total

categories = [
    ('Unseen\nAppearance', ood_appearance, n_ood_appearance),
    ('Unseen\nCombinations', ood_combinations, n_ood_combinations),
    ('Unseen\nObject Types', ood_obj_type, n_ood_type),
    ('OOD Average', ood_avg, n_ood_total),
]

n_categories = len(categories)

method_colors = [
    '#7BA3D0',   # VLA-Action
    '#5B8DC9',   # VLA-Vision
    '#B8D4F0',   # VLA-UniT w/o Cross-Recon
    '#F5D76E',   # VLA-UniT
]

def calc_se(p, n):
    return np.sqrt(p * (1 - p) / n)

fig, ax = plt.subplots(figsize=(10, 5))

group_width = n_methods + 1.5
bar_width = 0.75

for cat_idx, (cat_name, values, n_samples) in enumerate(categories):
    se = np.array([calc_se(p, n_samples) for p in values])
    base_x = cat_idx * group_width
    x_positions = base_x + np.arange(n_methods)

    shadow_offset_x = 0.10
    shadow_offset_y = -0.012
    shadow_values = np.maximum(values - 0.20 + shadow_offset_y, 0)
    ax.bar(x_positions + shadow_offset_x, shadow_values, bar_width,
           color='#AAAAAA', edgecolor='none', zorder=1, alpha=0.55,
           bottom=0.20)

    bars = ax.bar(x_positions, values - 0.20, bar_width, yerr=se,
                  color=method_colors,
                  edgecolor='#333333', linewidth=0.5, capsize=2,
                  error_kw={'elinewidth': 0.7, 'capthick': 0.7, 'ecolor': '#555555'},
                  zorder=2, bottom=0.20)

    best = np.argmax(values)
    for i, (bar, val, err) in enumerate(zip(bars, values, se)):
        y_pos = val + err + 0.010
        label = f'{val*100:.1f}'
        if i == best:
            ax.annotate(label, xy=(x_positions[i], y_pos), ha='center', va='bottom',
                        fontsize=8, fontweight='bold', color='#1A1A1A')
            bar.set_edgecolor('#1A1A1A')
            bar.set_linewidth(1.0)
        else:
            ax.annotate(label, xy=(x_positions[i], y_pos), ha='center', va='bottom',
                        fontsize=7.5, color='#555555')

category_centers = [cat_idx * group_width + (n_methods - 1) / 2 for cat_idx in range(n_categories)]
ax.set_xticks(category_centers)
ax.set_xticklabels([cat[0] for cat in categories], fontsize=10, fontweight='bold')

for cat_idx, (cat_name, values, n_samples) in enumerate(categories):
    base_x = cat_idx * group_width
    x_right = base_x + n_methods - 0.5
    ax.text(x_right, 0.68, f'n={n_samples}', ha='right', va='top',
            fontsize=7, color='#777777', style='italic')

ax.set_ylim(0.20, 0.70)
ax.set_yticks(np.arange(0.2, 0.75, 0.1))
ax.set_yticklabels([f'{int(x*100)}%' for x in np.arange(0.2, 0.75, 0.1)])
ax.set_ylabel('Success Rate', fontweight='bold')

ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#AAAAAA', linewidth=0.6)
ax.set_axisbelow(True)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_color('#555555')
ax.spines['left'].set_linewidth(0.6)
ax.spines['bottom'].set_color('#555555')
ax.spines['bottom'].set_linewidth(0.6)
ax.tick_params(bottom=False, left=True, colors='#333333', length=3, width=0.6)

ood_start = -0.8
ood_end = n_categories * group_width - 1
ax.axvspan(ood_start, ood_end, alpha=0.08, color='#F5D76E', zorder=0)

ax.set_xlim(ood_start, ood_end + 0.5)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=method_colors[i], edgecolor='#444444', linewidth=0.4, label=methods_display[i])
                   for i in range(n_methods)]
ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.18),
          ncol=4, frameon=False, handlelength=1.4, handletextpad=0.4,
          columnspacing=1.0, fontsize=10)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)

plt.savefig(f'{OUTPUT_DIR}ablation_ood.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(f'{OUTPUT_DIR}ablation_ood.pdf',
            bbox_inches='tight', facecolor='white', edgecolor='none')

print("ablation_ood.png/pdf")
