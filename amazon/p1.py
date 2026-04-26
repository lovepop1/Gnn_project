# ─────────────────────────────────────────────────────────────
# SEED BLOCK
# ─────────────────────────────────────────────────────────────
import torch, numpy as np, random, time, json, os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import (cohen_kappa_score, f1_score,
                             precision_score, recall_score)

torch.manual_seed(42); np.random.seed(42); random.seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ─────────────────────────────────────────────────────────────
# ENVIRONMENT INFO
# ─────────────────────────────────────────────────────────────
import sys
print(f"Python  version : {sys.version}")
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

try:
    import torch_geometric
    print(f"PyG     version : {torch_geometric.__version__}")
except ImportError:
    print("torch_geometric : NOT INSTALLED — run  !pip install torch_geometric")
    raise

# ─────────────────────────────────────────────────────────────
# MILESTONE 1 — DATASET LOAD & EDA
# ─────────────────────────────────────────────────────────────
from torch_geometric.datasets import Amazon
from torch_geometric.utils import to_networkx
import networkx as nx

print("\n" + "="*60)
print("MILESTONE 1 — Amazon Computers Dataset Analysis")
print("="*60)

# Load dataset
dataset = Amazon(root='.', name='Computers')
data    = dataset[0]

# ── 1. Core stats ─────────────────────────────────────────────
print("\n── 1. Core Statistics ──────────────────────────────────")
print(f"  num_nodes         : {data.num_nodes}")
print(f"  num_edges         : {data.num_edges}")
print(f"  num_node_features : {data.num_node_features}")
print(f"  num_classes       : {dataset.num_classes}")

# ── 2. Shape confirmation ─────────────────────────────────────
print("\n── 2. Shape Confirmation ───────────────────────────────")
x_ok  = data.x.shape          == torch.Size([13752, 767])
ei_ok = data.edge_index.shape == torch.Size([2, 245778])
print(f"  data.x.shape          == [13752, 767]   → {'✓ CONFIRMED' if x_ok  else '✗ MISMATCH: ' + str(data.x.shape)}")
print(f"  data.edge_index.shape == [2, 245778]    → {'✓ CONFIRMED' if ei_ok else '✗ MISMATCH: ' + str(data.edge_index.shape)}")

# ── 3. Feature sparsity ───────────────────────────────────────
print("\n── 3. Feature Sparsity ─────────────────────────────────")
total_elements = data.x.numel()
zero_elements  = (data.x == 0).sum().item()
sparsity       = zero_elements / total_elements
print(f"  Total feature values : {total_elements:,}")
print(f"  Zero  feature values : {zero_elements:,}")
print(f"  Sparsity (fraction=0): {sparsity:.6f}  ({sparsity*100:.2f}%)")

# ── 4. Class distribution ─────────────────────────────────────
print("\n── 4. Class Distribution ───────────────────────────────")
labels   = data.y.numpy()
n_nodes  = data.num_nodes
print(f"  {'Class':>6}  {'Count':>7}  {'Percent':>8}")
print(f"  {'─'*6}  {'─'*7}  {'─'*8}")
for cls in range(dataset.num_classes):
    cnt  = (labels == cls).sum()
    pct  = cnt / n_nodes * 100
    print(f"  {cls:>6}  {cnt:>7,}  {pct:>7.2f}%")

# ── 5. Degree statistics ──────────────────────────────────────
print("\n── 5. Degree Statistics ────────────────────────────────")
from torch_geometric.utils import degree
deg = degree(data.edge_index[0], num_nodes=data.num_nodes).numpy()
print(f"  Mean   degree : {deg.mean():.4f}")
print(f"  Median degree : {np.median(deg):.1f}")
print(f"  Max    degree : {deg.max():.0f}")
print(f"  Min    degree : {deg.min():.0f}")
print(f"  Std    degree : {deg.std():.4f}")

# ── 6. Top-10 highest-degree nodes ───────────────────────────
print("\n── 6. Top-10 Highest-Degree Nodes ──────────────────────")
top10_idx = np.argsort(deg)[::-1][:10]
print(f"  {'Rank':>4}  {'Node Index':>12}  {'Degree':>8}")
print(f"  {'─'*4}  {'─'*12}  {'─'*8}")
for rank, node_idx in enumerate(top10_idx, 1):
    print(f"  {rank:>4}  {node_idx:>12}  {int(deg[node_idx]):>8,}")

# ── 7. Edge attribute check ───────────────────────────────────
print("\n── 7. Edge Attribute Check ─────────────────────────────")
has_edge_attr = hasattr(data, 'edge_attr') and data.edge_attr is not None
if not has_edge_attr:
    print("  data.edge_attr does NOT exist → ✓ CONFIRMED (unweighted binary graph)")
else:
    print(f"  data.edge_attr EXISTS with shape {data.edge_attr.shape}  ← unexpected")

# ── 8. Graph connectivity ─────────────────────────────────────
print("\n── 8. Graph Connectivity ───────────────────────────────")
print("  Building NetworkX graph (may take a few seconds)…")
G          = to_networkx(data, to_undirected=True)
components = nx.number_connected_components(G)
print(f"  Number of connected components : {components}")
if components == 1:
    print("  → Graph is FULLY CONNECTED (one giant component)")
else:
    sizes = sorted([len(c) for c in nx.connected_components(G)], reverse=True)
    print(f"  → Largest component size : {sizes[0]:,} nodes")
    print(f"  → Component size summary : {sizes[:10]} …")

# ── Justification block ───────────────────────────────────────
print("\n" + "="*60)
print("WHY BOTH NODE FEATURES AND GRAPH STRUCTURE ARE NECESSARY:")
print("="*60)
print("""
  Node features alone (BoW from reviews): words like 'DDR4' and 'SATA'
  appear in both laptop AND desktop reviews — features are ambiguous across
  product categories. A feature-only MLP cannot disambiguate.

  Graph structure alone: a product with no review text cannot be classified
  from connectivity alone if its neighbours are equally ambiguous.

  Together: a generic 'Carrying Case' with ambiguous reviews is co-purchased
  with laptops → GNN assigns it to Computer Accessories. The same case
  co-purchased with cameras → Photo Accessories. Structure resolves ambiguity
  that features cannot, and features identify technical specs that structure
  cannot. Both are necessary and complementary.
""")

print("="*60)
print("MILESTONE 1 COMPLETE")
print("="*60)