"""
Teaser：Ablation 四方法的 OOD Average（与 plot_ablation.py 中 OOD 行统计一致）。
加权：Unseen Obj Appearance (18×50)、Unseen Obj-Container Combs (32×50)、
OOD Pick and Place Avg (23×50)；不修改 plot_ablation.py。
版式对齐 plot_teaser_real_robot / plot_teaser_method_comparison。
输出：teaser_ablation.png / .pdf
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
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.unicode_minus': False,
})

DATA_PATH = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/data/ablation.csv'
OUTPUT_DIR = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/'

# 与 plot_ablation 倒序后四方法配色一致：Vision → Action → w/o cross → Unified（黄）
BAR_COLORS = ['#5B8DC9', '#7BA3D0', '#B8D4F0', '#F5D76E']
HIGHLIGHT_IDX = 3  # Unified-token

INTER_BAR_CENTER = 1.0
BAR_WIDTH = 0.36
FIGSIZE_INCHES = (5.2, 3.85)
X_MARGIN = 0.40
SUBPLOT_BOTTOM = 0.32

N_OOD_APPEARANCE = 18 * 50
N_OOD_CONTAINER = 32 * 50
N_OOD_TYPE = 23 * 50
N_OOD_TOTAL = N_OOD_APPEARANCE + N_OOD_CONTAINER + N_OOD_TYPE


def calc_se(p, n):
    if p <= 0 or p >= 1:
        return 0.0
    return np.sqrt(p * (1.0 - p) / n)


def ood_average_row(row):
    """与 plot_ablation 中 OOD Average 相同：三 OOD 项按试验数加权。"""
    a = float(row['Unseen Obj Appearance'])
    c = float(row['Unseen Obj-Container Combs'])
    t = float(row['OOD Pick and Place Avg'])
    return (
        a * N_OOD_APPEARANCE + c * N_OOD_CONTAINER + t * N_OOD_TYPE
    ) / N_OOD_TOTAL


def main():
    df = pd.read_csv(DATA_PATH)
    # 与 plot_ablation 一致：倒序，Unified-token 最右
    df_rev = df.iloc[::-1].reset_index(drop=True)

    values = np.array([ood_average_row(df_rev.iloc[i]) for i in range(len(df_rev))], dtype=float)
    se = np.array([calc_se(p, N_OOD_TOTAL) for p in values])

    x_labels = [
        'VLA-Vision',
        'VLA-Action',
        'VLA-UniT\nw/o Cross Recon',
        'VLA-UniT',
    ]

    n_methods = len(values)
    base_y = 0.20
    heights = values - base_y

    x_positions = np.arange(n_methods, dtype=float) * INTER_BAR_CENTER
    bar_width = BAR_WIDTH

    fig, ax = plt.subplots(figsize=FIGSIZE_INCHES)

    shadow_offset_x = 0.02
    shadow_offset_y = -0.010
    shadow_values = np.maximum(values - base_y + shadow_offset_y, 0)
    ax.bar(
        x_positions + shadow_offset_x, shadow_values, bar_width,
        color='#000000', edgecolor='none', zorder=1, alpha=0.38,
        bottom=base_y,
    )

    bars = ax.bar(
        x_positions, heights, bar_width, yerr=se,
        color=BAR_COLORS, edgecolor='#333333', linewidth=0.5, capsize=1.2,
        error_kw={'elinewidth': 0.7, 'capthick': 0.7, 'ecolor': '#555555'},
        zorder=2, bottom=base_y,
    )

    for i, (bar, val, err) in enumerate(zip(bars, values, se)):
        y_pos = val + err + 0.010
        label = f'{val * 100:.1f}'
        if i == HIGHLIGHT_IDX:
            ax.annotate(
                label, xy=(x_positions[i], y_pos), ha='center', va='bottom',
                fontsize=10.5, fontweight='bold', color='#1A1A1A',
            )
            bar.set_edgecolor('#1A1A1A')
            bar.set_linewidth(1.35)
        else:
            ax.annotate(
                label, xy=(x_positions[i], y_pos), ha='center', va='bottom',
                fontsize=11.0, color='#555555',
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        x_labels, fontsize=10.0, fontweight='bold', color='#333333',
    )

    ax.set_xlim(x_positions[0] - X_MARGIN, x_positions[-1] + X_MARGIN)
    ax.set_ylim(0.30, 0.62)
    ax.set_yticks([])
    ax.yaxis.grid(False)

    # 仅轴名 + 上行箭头（与 EPE ↓ 对应），不画 y 轴刻度线
    ax.set_ylabel(
        'Success rate ↑', fontsize=10, fontweight='bold', color='#333333', labelpad=5,
    )

    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#888888')
    ax.spines['bottom'].set_linewidth(0.75)
    ax.tick_params(bottom=False, axis='y', left=False, labelleft=False)

    plt.subplots_adjust(
        bottom=SUBPLOT_BOTTOM, left=0.12, right=0.97, top=0.96,
    )

    out = OUTPUT_DIR + 'teaser_ablation'
    plt.savefig(out + '.png', dpi=300, facecolor='white', edgecolor='none')
    plt.savefig(out + '.pdf', facecolor='white', edgecolor='none')
    plt.close()
    print('Saved', out + '.png/pdf')


if __name__ == '__main__':
    main()
