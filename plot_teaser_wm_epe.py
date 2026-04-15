"""
Teaser 风格 EPE：两档 benchmark × Human / Humanoid 各一根柱。
x 轴仅两个刻度名：Human、Humanoid（各标在「档1 / 档2」一对 Raw+WM-UniT 的几何中心）。
图例：Raw Action WM / WM-UniT。y：EPE ↓。
作图规范与此前 WM EPE teaser 一致；勿用 bbox_inches='tight'。
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

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

COLOR_RAW = '#C5DCF2'
COLOR_UNIT = '#F5D76E'
LABEL_RAW = 'Raw Action WM'
LABEL_UNIT = 'WM-UniT'

INTER_BAR_CENTER = 0.72
BAR_WIDTH = 0.165
PAIR_OFFSET = 0.128

X_MARGIN_LEFT = 0.42
X_MARGIN_RIGHT = 0.26
_REF_XLIM = 1.0 + X_MARGIN_LEFT + X_MARGIN_RIGHT
_XLIM = INTER_BAR_CENTER + X_MARGIN_LEFT + X_MARGIN_RIGHT
# 略扁的纵向比例 + 抬高柱底，避免 ~0.7 的柱在嵌入时占满过高
FIGSIZE_INCHES = (4.85 * (_XLIM / _REF_XLIM), 2.85)
SUBPLOT_BOTTOM = 0.22
SUBPLOT_LEFT = 0.11
SUBPLOT_RIGHT = 0.97
SUBPLOT_TOP = 0.93

BASE_Y = 0.42
# 不压柱底；加大 y 轴上沿留白，使柱子在版面中占比更低、便于嵌入
Y_TOP_PAD = 0.22

# 两档 benchmark 上 Raw / WM-UniT 的 EPE（顺序：Human、Humanoid）
EPE_RAW_ACTION_WM = np.array([0.7060, 0.558], dtype=float)
EPE_WM_UNIT = np.array([0.5191, 0.453], dtype=float)

SHADOW_OFFSET_X = 0.02
SHADOW_OFFSET_Y = -0.010


def main():
    n = len(EPE_RAW_ACTION_WM)
    x_centers = np.arange(n, dtype=float) * INTER_BAR_CENTER
    x_raw = x_centers - PAIR_OFFSET
    x_unit = x_centers + PAIR_OFFSET

    fig, ax = plt.subplots(figsize=FIGSIZE_INCHES)
    base_y = BASE_Y

    for i in range(n):
        vr, vu = EPE_RAW_ACTION_WM[i], EPE_WM_UNIT[i]
        hr, hu = vr - base_y, vu - base_y

        s_r = np.maximum(hr + SHADOW_OFFSET_Y, 0)
        ax.bar(
            x_raw[i] + SHADOW_OFFSET_X, s_r, BAR_WIDTH,
            color='#000000', edgecolor='none', zorder=1, alpha=0.38,
            bottom=base_y,
        )
        ax.bar(
            x_raw[i], hr, BAR_WIDTH,
            color=COLOR_RAW, edgecolor='#333333', linewidth=0.5,
            zorder=2, bottom=base_y,
        )
        ax.annotate(
            f'{vr:.3f}', xy=(x_raw[i], vr + 0.006), ha='center', va='bottom',
            fontsize=8.0, color='#555555', clip_on=False,
        )

        s_u = np.maximum(hu + SHADOW_OFFSET_Y, 0)
        ax.bar(
            x_unit[i] + SHADOW_OFFSET_X, s_u, BAR_WIDTH,
            color='#000000', edgecolor='none', zorder=1, alpha=0.38,
            bottom=base_y,
        )
        ax.bar(
            x_unit[i], hu, BAR_WIDTH,
            color=COLOR_UNIT, edgecolor='#1A1A1A', linewidth=1.25,
            zorder=3, bottom=base_y,
        )
        ax.annotate(
            f'{vu:.3f}', xy=(x_unit[i], vu + 0.006), ha='center', va='bottom',
            fontsize=8.0, fontweight='bold', color='#1A1A1A', clip_on=False,
        )

    # Human / Humanoid 各对应一档 benchmark：刻度在「该档 Raw + WM-UniT」两柱的中点
    x_tick_pos = np.array([(x_raw[0] + x_unit[0]) / 2, (x_raw[1] + x_unit[1]) / 2])
    ax.set_xticks(x_tick_pos)
    ax.set_xticklabels(
        ['Human', 'Humanoid'],
        fontsize=10.0, fontweight='bold', color='#333333',
    )

    max_val = max(EPE_RAW_ACTION_WM.max(), EPE_WM_UNIT.max())
    y_top = max_val + Y_TOP_PAD
    ax.set_xlim(x_centers[0] - X_MARGIN_LEFT, x_centers[-1] + X_MARGIN_RIGHT)
    ax.set_ylim(base_y, y_top)

    ax.set_ylabel(
        'EPE',
        fontweight='bold',
        color='#333333',
        fontsize=9.0,
        labelpad=1,
    )
    ax.yaxis.set_label_coords(-0.06, 0.45)
    ax.text(-0.075, 0.62, '↓', transform=ax.transAxes,
            fontsize=16, fontweight='bold', color='#C62828',
            ha='center', va='center', rotation=0)

    ax.set_yticks([])
    ax.yaxis.grid(False)

    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#888888')
    ax.spines['bottom'].set_linewidth(0.75)
    ax.tick_params(axis='x', bottom=True, labelbottom=True, length=0, pad=2)
    ax.tick_params(axis='y', left=False, labelleft=False)

    legend_elements = [
        Patch(facecolor=COLOR_RAW, edgecolor='#333333', linewidth=0.5, label=LABEL_RAW),
        Patch(facecolor=COLOR_UNIT, edgecolor='#1A1A1A', linewidth=1.0, label=LABEL_UNIT),
    ]
    leg = ax.legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(0.985, 0.985),
        frameon=False,
        fontsize=8.0,
        handlelength=1.1,
        handletextpad=0.35,
        borderaxespad=0.2,
    )
    leg.set_zorder(10)

    plt.subplots_adjust(
        bottom=SUBPLOT_BOTTOM,
        left=SUBPLOT_LEFT,
        right=SUBPLOT_RIGHT,
        top=SUBPLOT_TOP,
    )

    out = OUTPUT_DIR + 'teaser_wm_epe'
    plt.savefig(out + '.png', dpi=300, facecolor='white', edgecolor='none')
    plt.savefig(out + '.pdf', facecolor='white', edgecolor='none')
    plt.close()
    print('Saved', out + '.png/pdf')


if __name__ == '__main__':
    main()
