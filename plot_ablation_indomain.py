"""
Ablation Study - In-Domain + OOD Transfer (Pick & Place)
验证不同 tokenizer 范式在 pick-and-place 迁移上的表现
"""

import matplotlib.pyplot as plt
import numpy as np

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

OUTPUT_DIR = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/'

methods = ['VLA-UniT w/o Cross-Recon', 'VLA-Action', 'VLA-Villa', 'VLA-Vision', 'VLA-UniT']
n_methods = len(methods)

pick_place = np.array([0.411, 0.570, 0.631, 0.636, 0.668])
unseen_comb = np.array([0.384, 0.591, 0.633, 0.631, 0.651])

n_pick = 18 * 50
n_comb = 23 * 50

def calc_se(p, n):
    return np.sqrt(p * (1 - p) / n)

se_pick = np.array([calc_se(p, n_pick) for p in pick_place])
se_comb = np.array([calc_se(p, n_comb) for p in unseen_comb])

method_colors = [
    '#B8D4F0',   # w/o Cross-Recon
    '#7BA3D0',   # VLA-Action
    '#A8C8E8',   # VLA-Villa
    '#5B8DC9',   # VLA-Vision
    '#F5D76E',   # VLA-UniT
]

fig, ax = plt.subplots(figsize=(6, 5))

categories = [
    ('Pick & Place\n(In-Domain)', pick_place, se_pick, n_pick),
]

group_width = n_methods + 1.5
bar_width = 0.75

for cat_idx, (cat_name, values, se, n_samples) in enumerate(categories):
    base_x = cat_idx * group_width
    x_positions = base_x + np.arange(n_methods)

    shadow_offset_x = 0.10
    shadow_values = np.maximum(values - 0.40, 0)
    ax.bar(x_positions + shadow_offset_x, shadow_values, bar_width,
           color='#AAAAAA', edgecolor='none', zorder=1, alpha=0.55,
           bottom=0.40)

    bars = ax.bar(x_positions, values - 0.40, bar_width, yerr=se,
                  color=method_colors,
                  edgecolor='#333333', linewidth=0.5, capsize=2,
                  error_kw={'elinewidth': 0.7, 'capthick': 0.7, 'ecolor': '#555555'},
                  zorder=2, bottom=0.40)

    best = np.argmax(values)
    for i, (bar, val, err) in enumerate(zip(bars, values, se)):
        y_pos = val + err + 0.008
        label = f'{val*100:.1f}'
        if i == best:
            ax.annotate(label, xy=(x_positions[i], y_pos), ha='center', va='bottom',
                        fontsize=8, fontweight='bold', color='#1A1A1A')
            bar.set_edgecolor('#1A1A1A')
            bar.set_linewidth(1.0)
        else:
            ax.annotate(label, xy=(x_positions[i], y_pos), ha='center', va='bottom',
                        fontsize=7.5, color='#555555')

category_centers = [cat_idx * group_width + (n_methods - 1) / 2 for cat_idx in range(len(categories))]
ax.set_xticks(category_centers)
ax.set_xticklabels([cat[0] for cat in categories], fontsize=10, fontweight='bold')

for cat_idx, (cat_name, values, se, n_samples) in enumerate(categories):
    base_x = cat_idx * group_width
    x_right = base_x + n_methods - 0.5
    ax.text(x_right, 0.73, f'n={n_samples}', ha='right', va='top',
            fontsize=7, color='#777777', style='italic')

ax.set_ylim(0.40, 0.75)
ax.set_yticks(np.arange(0.40, 0.76, 0.05))
ax.set_yticklabels([f'{int(x*100)}%' for x in np.arange(0.40, 0.76, 0.05)])
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

x_start = -0.8
x_end = n_methods - 0.2
ax.axvspan(x_start, x_end, ymin=0, ymax=1, alpha=0.08, color='#5B8DC9', zorder=0)
ax.set_xlim(x_start, x_end + 0.3)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=method_colors[i], edgecolor='#444444', linewidth=0.4, label=methods[i].replace('\n', ' '))
                   for i in range(n_methods)]
ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.25),
          ncol=3, frameon=False, handlelength=1.2, handletextpad=0.3,
          columnspacing=0.8, fontsize=8.5)

plt.tight_layout()
plt.subplots_adjust(bottom=0.24)

plt.savefig(f'{OUTPUT_DIR}ablation_indomain.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(f'{OUTPUT_DIR}ablation_indomain.pdf',
            bbox_inches='tight', facecolor='white', edgecolor='none')

print("ablation_indomain.png/pdf")
