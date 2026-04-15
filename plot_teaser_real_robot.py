"""
Teaser：真机 Overall（五类任务汇总成功数/总试验数），作图风格与 plot_teaser_method_comparison 对齐。
不修改 plot_real_robot.py；数据来自 data/real_robot.csv 前五列。
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

DATA_PATH = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/data/real_robot.csv'
OUTPUT_DIR = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/'

# 与 plot_real_robot 方法顺序、配色一致；最后一根为 w/ human（黄）
BAR_COLORS = ['#C5DCF2', '#7BA3D0', '#5B8DC9', '#F5D76E']
HIGHLIGHT_IDX = 3

# 与 plot_teaser_method_comparison 对齐
INTER_BAR_CENTER = 1.0
BAR_WIDTH = 0.36
FIGSIZE_INCHES = (5.2, 3.85)
X_MARGIN = 0.40
SUBPLOT_BOTTOM = 0.30

CATEGORY_COLS = [
    'Visual Robustness', 'Target Level Transfer', 'Distractor Robustness',
    'Geometric Generalization', 'Combinatorial Generalization',
]


def parse_fraction(frac_str):
    num, denom = frac_str.split('/')
    return int(num), int(denom)


def calc_se(p, n):
    return np.sqrt(p * (1.0 - p) / n)


def main():
    df = pd.read_csv(DATA_PATH)

    overall_p = []
    overall_n = []
    for _, row in df.iterrows():
        total_s, total_n = 0, 0
        for col in CATEGORY_COLS:
            s, n = parse_fraction(row[col])
            total_s += s
            total_n += n
        overall_p.append(total_s / total_n)
        overall_n.append(total_n)

    values = np.array(overall_p, dtype=float)
    se = np.array([calc_se(p, n) for p, n in zip(values, overall_n)])

    x_labels = [
        'GR00T-\nQwen2.5',
        'VLA-Vision',
        'VLA-UniT\nw/o Human',
        'VLA-UniT\nw/ Human Data',
    ]

    n_methods = len(values)
    # Overall 有低于 20% 的点，不用 base_y=0.2；从 0 起与 sim teaser 的「裁切 y」区分但柱画法一致
    base_y = 0.0

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
        x_positions, values - base_y, bar_width, yerr=se,
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
    ax.set_ylim(0.0, 0.78)
    ax.set_yticks([])
    ax.yaxis.grid(False)

    ax.set_ylabel(
        'Success rate', fontsize=10, fontweight='bold', color='#333333', labelpad=5,
    )
    ax.yaxis.set_label_coords(-0.08, 0.43)
    ax.text(-0.095, 0.70, '↑', transform=ax.transAxes,
            fontsize=16, fontweight='bold', color='#2E7D32',
            ha='center', va='center', rotation=0)

    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#888888')
    ax.spines['bottom'].set_linewidth(0.75)
    ax.tick_params(bottom=False, axis='y', left=False, labelleft=False)

    plt.subplots_adjust(
        bottom=SUBPLOT_BOTTOM, left=0.12, right=0.97, top=0.96,
    )

    out = OUTPUT_DIR + 'teaser_real_robot'
    plt.savefig(out + '.png', dpi=300, facecolor='white', edgecolor='none')
    plt.savefig(out + '.pdf', facecolor='white', edgecolor='none')
    plt.close()
    print('Saved', out + '.png/pdf')


if __name__ == '__main__':
    main()
