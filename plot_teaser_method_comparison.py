"""
Teaser: Unified-token (10%) → GR00T N1.6 → Unified-token full

柱间距（数据坐标）只由 INTER_BAR_CENTER、BAR_WIDTH 决定：
  相邻柱边净距 = INTER_BAR_CENTER - BAR_WIDTH
勿用 bbox_inches='tight' 保存，否则会裁掉留白，改 figsize 往往看不出导出变窄/变宽。

y 约 20%–72%；无 y 轴刻度/网格线；仅底边；阴影、误差棒、柱顶数值
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

EXPERIMENT_CSV = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/data/experiment_results.csv'
DATA_EFFICIENCY_CSV = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/data/data_efficiency.csv'
OUTPUT_DIR = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/'

# Unified 10% | GR00T | full Unified（黄与 plot_teaser_real_robot / real_robot highlight 一致）
BAR_COLORS = ['#B8D6EB', '#A8C8E6', '#F5D76E']

N_OVERALL = 24 * 50
HIGHLIGHT_IDX = 2  # full Unified-token: bold label + edge

# 柱布局（数据坐标）：改柱间疏密只动下面两个 + X_MARGIN
INTER_BAR_CENTER = 1.0  # 相邻柱心距离；增大 → 柱之间更疏
BAR_WIDTH = 0.36  # 柱宽；减小 → 柱更细、柱间更疏
FIGSIZE_INCHES = (5.2, 3.85)  # 与 plot_teaser_real_robot 画布宽对齐
X_MARGIN = 0.40
# 仅给两行 x 标签留高；过大 → 标签下方大块白（整幅保存时尤其明显）
SUBPLOT_BOTTOM = 0.26


def main():
    df_exp = pd.read_csv(EXPERIMENT_CSV).set_index('Method')
    df_eff = pd.read_csv(DATA_EFFICIENCY_CSV)

    v_groot = float(df_exp.loc['GROOT N1.6']['Overall'])
    row_10 = df_eff[
        (df_eff['Method'] == 'Unified-token')
        & (df_eff['Data Scale'] == '24×100 trajs')
    ]
    row_full = df_eff[
        (df_eff['Method'] == 'Unified-token')
        & (df_eff['Data Scale'] == '24×1000 trajs')
    ]
    v_unified_10 = float(row_10['Overall'].values[0])
    v_unified_full = float(row_full['Overall'].values[0])

    values = np.array([v_unified_10, v_groot, v_unified_full], dtype=float)
    se = np.sqrt(values * (1.0 - values) / N_OVERALL)

    x_labels = [
        'VLA-UniT\n(10% data)',
        'GR00T N1.6',
        'VLA-UniT',
    ]

    n_methods = 3
    base_y = 0.20

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

    # 纵轴目标 20–70%；顶略高于 70% 以免柱顶数字与误差棒上沿被裁切
    ax.set_ylim(0.20, 0.72)
    ax.set_yticks([])
    ax.yaxis.grid(False)

    ax.set_ylabel(
        'Success rate ↑', fontsize=10, fontweight='bold', color='#333333', labelpad=5,
    )

    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#888888')
    ax.spines['bottom'].set_linewidth(0.75)
    ax.tick_params(bottom=False, axis='y', left=False, labelleft=False)

    # 勿与 tight_layout 同用，易重复留白；只用手动边距
    plt.subplots_adjust(
        bottom=SUBPLOT_BOTTOM, left=0.12, right=0.97, top=0.96,
    )

    out = OUTPUT_DIR + 'teaser_method_comparison'
    # 默认 bbox_inches=None：整幅 figure 参与导出，figsize 才对应 PNG/PDF 实际宽度
    plt.savefig(out + '.png', dpi=300, facecolor='white', edgecolor='none')
    plt.savefig(out + '.pdf', facecolor='white', edgecolor='none')
    plt.close()
    print('Saved', out + '.png/pdf')


if __name__ == '__main__':
    main()
