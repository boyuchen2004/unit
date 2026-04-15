"""
t-SNE Visualization for Latent Action Distribution
统一蓝黄配色 + 采样优化 + 交替Z-order绘制
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
DATA_PATH = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/data/tsne/latent_action_distribution.npz'
OUTPUT_DIR = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/'

# ============== 采样参数 ==============
N_SAMPLES = 500  # 每类采样点数，减少点数使图更紧致

# ============== 加载数据 ==============
data = np.load(DATA_PATH)
pooled_a_full = data['pooled_quant_a']  # GR1_EEF quantized: (2000, 32)
pooled_b_full = data['pooled_quant_b']  # EgoDex quantized: (2000, 32)
# 随机采样
np.random.seed(42)
idx_a = np.random.choice(len(pooled_a_full), N_SAMPLES, replace=False)
idx_b = np.random.choice(len(pooled_b_full), N_SAMPLES, replace=False)
pooled_a = pooled_a_full[idx_a]
pooled_b = pooled_b_full[idx_b]

print(f"Dataset A: {pooled_a.shape} (sampled from {len(pooled_a_full)})")
print(f"Dataset B: {pooled_b.shape} (sampled from {len(pooled_b_full)})")

# ============== 标准化 + PCA + t-SNE 降维 ==============
combined = np.vstack([pooled_a, pooled_b])
labels = np.array([0] * len(pooled_a) + [1] * len(pooled_b))

print("Standardizing...")
scaler = StandardScaler()
combined_scaled = scaler.fit_transform(combined)

print("Running PCA...")
pca = PCA(n_components=8, random_state=42)
combined_pca = pca.fit_transform(combined_scaled)
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

print("Running t-SNE with cosine metric...")
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    init='random',      # cosine不支持pca init
    random_state=42,
    max_iter=2000,
    early_exaggeration=12,
    metric='cosine',
)
embedded = tsne.fit_transform(combined_pca)

# 分离两类数据
emb_a = embedded[labels == 0]
emb_b = embedded[labels == 1]

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

# 一次性绘制所有点（按交替顺序）
ax.scatter(plot_x, plot_y, c=plot_colors, 
           s=50,           # 较大的点
           alpha=0.8,      # 适中透明度
           edgecolors='none',  # 无边缘，更柔和
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

ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# 添加轻微padding
x_margin = (plot_x.max() - plot_x.min()) * 0.05
y_margin = (plot_y.max() - plot_y.min()) * 0.05
ax.set_xlim(plot_x.min() - x_margin, plot_x.max() + x_margin)
ax.set_ylim(plot_y.min() - y_margin, plot_y.max() + y_margin)

plt.tight_layout()

# ============== 保存 ==============
plt.savefig(f'{OUTPUT_DIR}tsne_latent_action.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig(f'{OUTPUT_DIR}tsne_latent_action.pdf', 
            bbox_inches='tight', facecolor='white', edgecolor='none')

print("✓ Saved: tsne_latent_action.png/pdf")
print(f"  - {LABEL_A}: {n_a}, {LABEL_B}: {n_b}")
