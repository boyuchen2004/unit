"""Raw vs Latent t-SNE + KDE.

对应脚本与数据:
  - raw:   plot_tsne_raw_action.py  -> data/tsne/raw_action_features.npz
  - latent: plot_tsne_latent_action.py -> data/tsne/latent_action_distribution.npz
图例: Humanoid = actions_a / pooled_a (GR1), Human = actions_b / pooled_b (EgoDex).
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/'
PATH_RAW = OUTPUT_DIR + 'data/tsne/raw_action_features.npz'
PATH_LATENT = OUTPUT_DIR + 'data/tsne/latent_action_distribution.npz'
# 莫兰迪：与论文其它图一致，柔和、不刺眼
COLOR_HUMANOID = '#5B8DC9'
COLOR_HUMAN = '#7BA38A'
N_SAMPLES = 2000  # 每类；与 plot_tsne_raw_action 的 t-SNE 拟合规模一致（绘图脚本再子采样到 500/类）
RNG_SAMPLE = np.random.default_rng(42)
RNG_SHUFFLE = np.random.RandomState(123)


def tsne_raw_action(pooled_a, pooled_b):
    """与 plot_tsne_raw_action.py 一致: StandardScaler, PCA-16, TSNE cosine。"""
    combined = np.vstack([pooled_a, pooled_b])
    labels = np.array([0] * len(pooled_a) + [1] * len(pooled_b))
    scaler = StandardScaler()
    combined_pca = PCA(n_components=16, random_state=42).fit_transform(
        scaler.fit_transform(combined)
    )
    tsne = TSNE(
        n_components=2,
        perplexity=32,
        learning_rate='auto',
        init='pca',
        random_state=42,
        max_iter=3000,
        early_exaggeration=22,
        metric='cosine',
    )
    return tsne.fit_transform(combined_pca), labels


def tsne_latent_action(pooled_a, pooled_b):
    """与 plot_tsne_latent_action.py 一致: quantized, PCA-8, TSNE cosine."""
    combined = np.vstack([pooled_a, pooled_b])
    labels = np.array([0] * len(pooled_a) + [1] * len(pooled_b))
    scaler = StandardScaler()
    combined_pca = PCA(n_components=8, random_state=42).fit_transform(
        scaler.fit_transform(combined)
    )
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate='auto',
        init='random',
        random_state=42,
        max_iter=2000,
        early_exaggeration=12,
        metric='cosine',
    )
    return tsne.fit_transform(combined_pca), labels


def kde_grid(z, xlim, ylim, n=80):
    xi = np.linspace(xlim[0], xlim[1], n)
    yi = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(xi, yi)
    kde = gaussian_kde(z.T)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    return X, Y, Z


def kde_grid_combined(ea, eb, xlim, ylim, n=120):
    z = np.vstack([ea, eb])
    return kde_grid(z, xlim, ylim, n=n)


def lims(ea, eb, pad=0.08):
    x = np.concatenate([ea[:, 0], eb[:, 0]])
    y = np.concatenate([ea[:, 1], eb[:, 1]])
    xr = x.max() - x.min()
    yr = y.max() - y.min()
    return (x.min() - pad * xr, x.max() + pad * xr), (y.min() - pad * yr, y.max() + pad * yr)


def _density_for_plot(Z, rel_floor=0.20, pct_floor=32):
    """归一化到 [0,1]；低于峰值 rel_floor 且低于分位 pct_floor 的区域不填色（白底）。"""
    zmax = Z.max() + 1e-12
    Z = Z / zmax
    thr = max(rel_floor, np.percentile(Z, pct_floor))
    Zp = np.where(Z < thr, np.nan, Z)
    return Zp, float(thr)


def blend_panel(ax, ea, eb, names, title, xlim, ylim):
    """单层浅灰密度（全体点）+ 交替散点；避免双色密度叠涂显脏。"""
    X, Y, Z = kde_grid_combined(ea, eb, xlim, ylim, n=120)
    Z, thr = _density_for_plot(Z, rel_floor=0.18, pct_floor=30)
    nlev = 14
    levels = np.linspace(thr, 1.0, nlev + 1)
    ax.contourf(
        X, Y, Z, levels=levels,
        colors=plt.cm.Greys(np.linspace(0.9, 0.65, nlev)),
        alpha=0.55,
        antialiased=True,
        zorder=1,
    )

    na, nb = len(ea), len(eb)
    ia, ib = np.arange(na), np.arange(nb)
    RNG_SHUFFLE.shuffle(ia)
    RNG_SHUFFLE.shuffle(ib)
    xs, ys, cs = [], [], []
    m = min(na, nb)
    for i in range(m):
        xs += [ea[ia[i], 0], eb[ib[i], 0]]
        ys += [ea[ia[i], 1], eb[ib[i], 1]]
        cs += [COLOR_HUMANOID, COLOR_HUMAN]
    for i in range(m, na):
        xs.append(ea[ia[i], 0])
        ys.append(ea[ia[i], 1])
        cs.append(COLOR_HUMANOID)
    for i in range(m, nb):
        xs.append(eb[ib[i], 0])
        ys.append(eb[ib[i], 1])
        cs.append(COLOR_HUMAN)

    ax.scatter(
        xs, ys, c=cs, s=34, alpha=0.82,
        edgecolors='white', linewidths=0.35,
        zorder=5,
    )
    ax.scatter([], [], c=COLOR_HUMANOID, s=52, label=names[0],
               edgecolors='white', linewidths=0.35)
    ax.scatter([], [], c=COLOR_HUMAN, s=52, label=names[1],
               edgecolors='white', linewidths=0.35)

    ax.set_title(title, fontsize=12, color='#2C2C2C', pad=8)
    ax.set_xlabel('t-SNE dimension 1', fontsize=10, color='#555555')
    ax.set_ylabel('t-SNE dimension 2', fontsize=10, color='#555555')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        loc='upper right', frameon=False, fontsize=9,
        labelcolor='#444444',
    )
    ax.set_facecolor('white')
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#AAAAAA')
    ax.spines['bottom'].set_color('#AAAAAA')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def main():
    dr = np.load(PATH_RAW)
    ra = dr['actions_a'].astype(np.float32)
    rb = dr['actions_b'].astype(np.float32)
    ia = RNG_SAMPLE.choice(len(ra), N_SAMPLES, replace=False)
    ib = RNG_SAMPLE.choice(len(rb), N_SAMPLES, replace=False)
    emb_r, lab_r = tsne_raw_action(ra[ia], rb[ib])
    ea_r, eb_r = emb_r[lab_r == 0], emb_r[lab_r == 1]

    dl = np.load(PATH_LATENT)
    la = dl['pooled_quant_a']
    lb = dl['pooled_quant_b']
    ila = RNG_SAMPLE.choice(len(la), N_SAMPLES, replace=False)
    ilb = RNG_SAMPLE.choice(len(lb), N_SAMPLES, replace=False)
    emb_l, lab_l = tsne_latent_action(la[ila], lb[ilb])
    ea_l, eb_l = emb_l[lab_l == 0], emb_l[lab_l == 1]

    # A=GR1/humanoid, B=EgoDex/human
    names = ('Humanoid', 'Human')
    xl_r, yl_r = lims(ea_r, eb_r)
    xl_l, yl_l = lims(ea_l, eb_l)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 10))
    fig.patch.set_facecolor('white')
    blend_panel(axes[0], ea_r, eb_r, names, 'Raw action', xl_r, yl_r)
    blend_panel(axes[1], ea_l, eb_l, names, 'Unified latent action', xl_l, yl_l)
    fig.suptitle(
        'Raw vs unified latent action',
        fontsize=13,
        color='#2C2C2C',
        y=1.02,
    )
    plt.tight_layout()
    out = OUTPUT_DIR + 'tsne_raw_vs_latent_density'
    plt.savefig(out + '.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(out + '.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved', out + '.png/pdf')


if __name__ == '__main__':
    main()
