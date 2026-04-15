"""Export all 6 t-SNE datasets to JSON for HTML visualization.
Logic: run t-SNE on full dataset, then subsample for display.
"""
import json, numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

OUT = '/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/data/tsne_coords.json'
N_DISPLAY = 600  # per class for display (after t-SNE)

def sample(arr, n, seed=42):
    rng = np.random.RandomState(seed)
    if n >= len(arr): return arr
    return arr[rng.choice(len(arr), n, replace=False)]

def subsample_emb(emb, n, seed=789):
    rng = np.random.RandomState(seed)
    if n >= len(emb): return emb
    return emb[rng.choice(len(emb), n, replace=False)]

result = {}

# 1. Raw Action
print("1/6: Raw Action...")
d = np.load('/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/data/tsne/raw_action_features.npz')
a = sample(d['actions_a'].astype(np.float32), 2000)
b = sample(d['actions_b'].astype(np.float32), 2000)
combined = np.vstack([a, b])
labels = np.array([0]*len(a) + [1]*len(b))
sc = StandardScaler().fit_transform(combined)
pc = PCA(n_components=16, random_state=42).fit_transform(sc)
tsne = TSNE(n_components=2, perplexity=32, learning_rate='auto', init='pca',
            random_state=42, max_iter=3000, early_exaggeration=22, metric='cosine')
emb = tsne.fit_transform(pc)
emb_a = subsample_emb(emb[labels==0], N_DISPLAY)
emb_b = subsample_emb(emb[labels==1], N_DISPLAY)
result['raw_action'] = {'humanoid': emb_a.tolist(), 'human': emb_b.tolist()}

# 2. Latent Action (UniT tokens)
print("2/6: Latent Action...")
d = np.load('/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/data/tsne/latent_action_distribution.npz')
a = d['pooled_quant_a']  # full 2000
b = d['pooled_quant_b']  # full 2000
combined = np.vstack([a, b])
labels = np.array([0]*len(a) + [1]*len(b))
sc = StandardScaler().fit_transform(combined)
pc = PCA(n_components=8, random_state=42).fit_transform(sc)
tsne = TSNE(n_components=2, perplexity=10, learning_rate='auto', init='random',
            random_state=42, max_iter=2000, early_exaggeration=12, metric='cosine')
emb = tsne.fit_transform(pc)
emb_a = subsample_emb(emb[labels==0], N_DISPLAY)
emb_b = subsample_emb(emb[labels==1], N_DISPLAY)
result['latent_action'] = {'humanoid': emb_a.tolist(), 'human': emb_b.tolist()}

# 3. VL Features baseline
print("3/6: VL Features baseline...")
d = np.load('/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/data/tsne/vl_features_gr00t_baseline.npz')
a, b = d['pooled_a'], d['pooled_b']  # full 2000 each
combined = np.vstack([a, b])
labels = np.array([0]*len(a) + [1]*len(b))
tsne = TSNE(n_components=2, perplexity=10, max_iter=2000, init='pca', random_state=42)
emb = tsne.fit_transform(combined)
emb_a = subsample_emb(emb[labels==0], N_DISPLAY)
emb_b = subsample_emb(emb[labels==1], N_DISPLAY)
result['vl_baseline'] = {'humanoid': emb_a.tolist(), 'human': emb_b.tolist()}

# 4. VL Features UniT
print("4/6: VL Features UniT...")
d = np.load('/dataset_rc_mm/chenby10@xiaopeng.com/paper_plot/data/tsne/vl_features_unified_latent_action.npz')
a, b = d['pooled_a'], d['pooled_b']  # full 2000 each
combined = np.vstack([a, b])
labels = np.array([0]*len(a) + [1]*len(b))
tsne = TSNE(n_components=2, perplexity=100, max_iter=2000, init='pca', random_state=42)
emb = tsne.fit_transform(combined)
emb_a = subsample_emb(emb[labels==0], N_DISPLAY)
emb_b = subsample_emb(emb[labels==1], N_DISPLAY)
result['vl_unit'] = {'humanoid': emb_a.tolist(), 'human': emb_b.tolist()}

# 5. WM baseline
print("5/6: WM baseline...")
d = np.load('/dataset_rc_mm/chenby10@xiaopeng.com/Cosmos_predict_2.5/results/tsne_analysis/ac_multi_embodiment_no_latent_action_iter40000/ac_multi_embodiment_no_latent_action_crossattn_block_27_tsne_data.npz')
a, b = d['features_2'], d['features_3']  # full 2000 each
combined = np.vstack([a, b])
labels = np.array([0]*len(a) + [1]*len(b))
tsne = TSNE(n_components=2, perplexity=30, max_iter=2000, init='pca', random_state=42)
emb = tsne.fit_transform(combined)
emb_a = subsample_emb(emb[labels==0], N_DISPLAY)
emb_b = subsample_emb(emb[labels==1], N_DISPLAY)
result['wm_baseline'] = {'humanoid': emb_a.tolist(), 'human': emb_b.tolist()}

# 6. WM UniT
print("6/6: WM UniT...")
d = np.load('/dataset_rc_mm/chenby10@xiaopeng.com/Cosmos_predict_2.5/results/tsne_analysis/ac_multi_embodiment_latent_action_only_iter40000/ac_multi_embodiment_latent_action_only_crossattn_block_27_tsne_data.npz')
a, b = d['features_2'], d['features_3']  # full 2000 each
combined = np.vstack([a, b])
labels = np.array([0]*len(a) + [1]*len(b))
tsne = TSNE(n_components=2, perplexity=30, max_iter=2000, init='pca', random_state=42)
emb = tsne.fit_transform(combined)
emb_a = subsample_emb(emb[labels==0], N_DISPLAY)
emb_b = subsample_emb(emb[labels==1], N_DISPLAY)
result['wm_unit'] = {'humanoid': emb_a.tolist(), 'human': emb_b.tolist()}

with open(OUT, 'w') as f:
    json.dump(result, f)
print(f"Done: {OUT}")
