"""
t-SNE Visualization for Raw Action Features
统一蓝黄配色 + 交替Z-order绘制

流程：每类 2000 点（共 4000）拟合 StandardScaler+PCA+t-SNE；再在嵌入中每类随机采 500 点（共 1000）绘图。
t-SNE 在默认风格上略作调参（稍增 PCA 维、略降 perplexity、略提 early_exaggeration、cosine），略增两类可分性。
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============== 论文级别配置 ==============
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.unicode_minus': False,
})

# ============== 统一蓝黄配色 (与其他图一致) ==============
COLOR_BLUE = '#5B8DC9'    # 柔蓝
COLOR_YELLOW = '#F5D76E'  # 柔黄
LABEL_A = 'Humanoid'
LABEL_B = 'Human'

# ============== 数据路径 ==============
DATA_PATH = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/data/tsne/raw_action_features.npz'
OUTPUT_DIR = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/'

# ============== 采样 ==============
# t-SNE 拟合用：每类点数（合计 4000）
N_SAMPLES_TSNE = 2000
# 绘图层：每类点数（合计 1000，与原先一致）
N_SAMPLES_PLOT = 500
RANDOM_STATE_DATA = 42       # 从原始池取 2000/类
RANDOM_STATE_PLOT_SUB = 456  # 在 4000 点嵌入上取 500/类用于绘图

# ============== 降维 / t-SNE（轻度 trick：略增区分度）==============
PCA_N_COMPONENTS = 16
TSNE_PERPLEXITY = 32
TSNE_EARLY_EXAGGERATION = 22
TSNE_MAX_ITER = 3000
TSNE_METRIC = 'cosine'

# ============== 加载数据 ==============
data = np.load(DATA_PATH)

pooled_a_full = data['actions_a'].astype(np.float32)  # (5000, 2048)
pooled_b_full = data['actions_b'].astype(np.float32)  # (5000, 2048)

np.random.seed(RANDOM_STATE_DATA)
if N_SAMPLES_TSNE >= len(pooled_a_full):
    pooled_a = pooled_a_full
    pooled_b = pooled_b_full
else:
    idx_a = np.random.choice(len(pooled_a_full), N_SAMPLES_TSNE, replace=False)
    idx_b = np.random.choice(len(pooled_b_full), N_SAMPLES_TSNE, replace=False)
    pooled_a = pooled_a_full[idx_a]
    pooled_b = pooled_b_full[idx_b]

print(f"t-SNE fit: A {pooled_a.shape}, B {pooled_b.shape} (from {len(pooled_a_full)})")

# ============== 标准化 + PCA + t-SNE 降维 ==============
combined = np.vstack([pooled_a, pooled_b])
labels = np.array([0] * len(pooled_a) + [1] * len(pooled_b))

print("Standardizing...")
scaler = StandardScaler()
combined_scaled = scaler.fit_transform(combined)

print("Running PCA...")
pca = PCA(n_components=PCA_N_COMPONENTS, random_state=42)
combined_pca = pca.fit_transform(combined_scaled)
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

print(f"Running t-SNE (metric={TSNE_METRIC})...")
tsne = TSNE(
    n_components=2,
    perplexity=TSNE_PERPLEXITY,
    learning_rate='auto',
    init='pca',
    random_state=42,
    max_iter=TSNE_MAX_ITER,
    early_exaggeration=TSNE_EARLY_EXAGGERATION,
    metric=TSNE_METRIC,
)
embedded = tsne.fit_transform(combined_pca)

emb_a_full = embedded[labels == 0]
emb_b_full = embedded[labels == 1]

rng_sub = np.random.RandomState(RANDOM_STATE_PLOT_SUB)
idx_a = rng_sub.choice(len(emb_a_full), N_SAMPLES_PLOT, replace=False)
idx_b = rng_sub.choice(len(emb_b_full), N_SAMPLES_PLOT, replace=False)
emb_a = emb_a_full[idx_a]
emb_b = emb_b_full[idx_b]

print(f"Plot: {N_SAMPLES_PLOT}/class from embedding (total {2 * N_SAMPLES_PLOT})")

# ============== 交替绘制准备 (Shuffle Z-order) ==============
n_a, n_b = len(emb_a), len(emb_b)

np.random.seed(123)
indices_a = np.arange(n_a)
indices_b = np.arange(n_b)
np.random.shuffle(indices_a)
np.random.shuffle(indices_b)

# 交替合并两类数据点的坐标和标签
plot_x = []
plot_y = []
plot_colors = []

min_len = min(n_a, n_b)
for i in range(min_len):
    plot_x.append(emb_a[indices_a[i], 0])
    plot_y.append(emb_a[indices_a[i], 1])
    plot_colors.append(COLOR_BLUE)

    plot_x.append(emb_b[indices_b[i], 0])
    plot_y.append(emb_b[indices_b[i], 1])
    plot_colors.append(COLOR_YELLOW)

for i in range(min_len, n_a):
    plot_x.append(emb_a[indices_a[i], 0])
    plot_y.append(emb_a[indices_a[i], 1])
    plot_colors.append(COLOR_BLUE)
for i in range(min_len, n_b):
    plot_x.append(emb_b[indices_b[i], 0])
    plot_y.append(emb_b[indices_b[i], 1])
    plot_colors.append(COLOR_YELLOW)

plot_x = np.array(plot_x)
plot_y = np.array(plot_y)

# ============== 绘图 ==============
fig, ax = plt.subplots(figsize=(7, 6))

point_size = 20 if n_a > 1000 else 50
ax.scatter(plot_x, plot_y, c=plot_colors,
           s=point_size,
           alpha=0.7,
           edgecolors='none',
           zorder=2)

ax.scatter([], [], c=COLOR_BLUE, s=80, alpha=0.9,
           edgecolors='none', label=LABEL_A)
ax.scatter([], [], c=COLOR_YELLOW, s=80, alpha=0.9,
           edgecolors='none', label=LABEL_B)

# ============== 样式设置 ==============
ax.set_xlabel('t-SNE Dimension 1', fontweight='bold', color='#333333')
ax.set_ylabel('t-SNE Dimension 2', fontweight='bold', color='#333333')

# 隐藏刻度值
ax.set_xticks([])
ax.set_yticks([])

# 边框样式 - 只保留左和下
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_color('#888888')
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_color('#888888')
ax.spines['bottom'].set_linewidth(0.8)

# 背景纯白
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# 添加轻微padding
x_margin = (plot_x.max() - plot_x.min()) * 0.05
y_margin = (plot_y.max() - plot_y.min()) * 0.05
ax.set_xlim(plot_x.min() - x_margin, plot_x.max() + x_margin)
ax.set_ylim(plot_y.min() - y_margin, plot_y.max() + y_margin)

plt.tight_layout()

# ============== 保存 ==============
plt.savefig(f'{OUTPUT_DIR}tsne_raw_action.png',
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(f'{OUTPUT_DIR}tsne_raw_action.pdf',
            bbox_inches='tight', facecolor='white', edgecolor='none')

print("✓ Saved: tsne_raw_action.png/pdf")
print(f"  - {LABEL_A}: {n_a}, {LABEL_B}: {n_b} (plotted; t-SNE fit on {len(emb_a_full) + len(emb_b_full)})")
