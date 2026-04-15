"""
Teaser：WM-UniT Human pretrain 消融（EPE，越低越好）。
作图规范与 plot_teaser_real_robot / plot_teaser_wm_epe 一致：INTER_BAR_CENTER、BAR_WIDTH、
阴影、仅底轴、无刻度 y、EPE ↓ 轴名、柱顶数值；勿用 bbox_inches='tight'。
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
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.unicode_minus': False,
})

OUTPUT_DIR = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/'

INTER_BAR_CENTER = 1.0
BAR_WIDTH = 0.36
FIGSIZE_INCHES = (4.9, 3.85)
X_MARGIN = 0.40
SUBPLOT_BOTTOM = 0.30

# 浅蓝 | 黄强调（w/ pretrain 更优）
BAR_COLORS = ['#C5DCF2', '#F5D76E']
HIGHLIGHT_IDX = 1

# 抬高纵轴起点，压缩 EPE 显示区间，使 0.478 vs 0.446 的相对落差更醒目
BASE_Y = 0.432

EPE_WO = 0.478
EPE_W = 0.446

SHADOW_OFFSET_X = 0.02
SHADOW_OFFSET_Y = -0.010


def main():
    values = np.array([EPE_WO, EPE_W], dtype=float)
    x_labels = [
        'WM-UniT\nw/o Human pretrain',
        'WM-UniT\nw/ Human pretrain',
    ]

    n_methods = len(values)
    x_positions = np.arange(n_methods, dtype=float) * INTER_BAR_CENTER
    bar_width = BAR_WIDTH

    fig, ax = plt.subplots(figsize=FIGSIZE_INCHES)

    shadow_offset_x = SHADOW_OFFSET_X
    shadow_offset_y = SHADOW_OFFSET_Y
    shadow_values = np.maximum(values - BASE_Y + shadow_offset_y, 0)
    ax.bar(
        x_positions + shadow_offset_x, shadow_values, bar_width,
        color='#000000', edgecolor='none', zorder=1, alpha=0.38,
        bottom=BASE_Y,
    )

    bars = ax.bar(
        x_positions, values - BASE_Y, bar_width,
        color=BAR_COLORS, edgecolor='#333333', linewidth=0.5,
        zorder=2, bottom=BASE_Y,
    )

    for i, (bar, val) in enumerate(zip(bars, values)):
        y_pos = val + 0.006
        label = f'{val:.3f}'
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
        x_labels, fontsize=9.5, fontweight='bold', color='#333333',
    )

    ax.set_xlim(x_positions[0] - X_MARGIN, x_positions[-1] + X_MARGIN)
    y_top = max(values) + 0.028
    ax.set_ylim(BASE_Y, y_top)

    ax.set_ylabel(
        'EPE',
        fontweight='bold',
        color='#333333',
        fontsize=10.5,
        labelpad=2,
    )
    ax.yaxis.set_label_coords(-0.07, 0.45)
    ax.text(-0.085, 0.62, '↓', transform=ax.transAxes,
            fontsize=16, fontweight='bold', color='#C62828',
            ha='center', va='center', rotation=0)

    ax.set_yticks([])
    ax.yaxis.grid(False)

    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#888888')
    ax.spines['bottom'].set_linewidth(0.75)
    ax.tick_params(bottom=False, axis='y', left=False, labelleft=False)

    plt.subplots_adjust(
        bottom=SUBPLOT_BOTTOM, left=0.14, right=0.97, top=0.96,
    )

    out = OUTPUT_DIR + 'teaser_wm_human_pretrain'
    plt.savefig(out + '.png', dpi=300, facecolor='white', edgecolor='none')
    plt.savefig(out + '.pdf', facecolor='white', edgecolor='none')
    plt.close()
    print('Saved', out + '.png/pdf')


if __name__ == '__main__':
    main()
