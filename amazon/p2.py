# ─────────────────────────────────────────────────────────────
# INSTALL (run this cell first in Colab)
# !pip install torch_geometric -q
# ─────────────────────────────────────────────────────────────

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

import sys
import torch_geometric
print(f"Python  version : {sys.version}")
print(f"PyTorch version : {torch.__version__}")
print(f"PyG     version : {torch_geometric.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device          : {device}")

# ─────────────────────────────────────────────────────────────
# RESULTS SAVER
# ─────────────────────────────────────────────────────────────
def save_results(dataset_key, model_name, metrics):
    path = 'results.json'
    data = json.load(open(path)) if os.path.exists(path) else {}
    if dataset_key not in data: data[dataset_key] = {}
    data[dataset_key][model_name] = metrics
    json.dump(data, open(path, 'w'), indent=2)

# ─────────────────────────────────────────────────────────────
# MILESTONE 2 — LOAD DATASET & PREPARE SPLITS
# ─────────────────────────────────────────────────────────────
from torch_geometric.datasets import Amazon
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split

print("\n" + "="*65)
print("MILESTONE 2 — Amazon Computers: GNN Empirical Comparison")
print("="*65)

dataset = Amazon(root='.', name='Computers')
data    = dataset[0]

# Convert directed → undirected
data.edge_index = to_undirected(data.edge_index)
print(f"\nEdges after to_undirected: {data.num_edges:,}")

# ── Stratified 60 / 20 / 20 splits ───────────────────────────
labels   = data.y.numpy()
all_idx  = np.arange(data.num_nodes)

train_idx, temp_idx = train_test_split(
    all_idx, test_size=0.40, stratify=labels[all_idx], random_state=42)
val_idx, test_idx   = train_test_split(
    temp_idx, test_size=0.50, stratify=labels[temp_idx], random_state=42)

train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
val_mask   = torch.zeros(data.num_nodes, dtype=torch.bool)
test_mask  = torch.zeros(data.num_nodes, dtype=torch.bool)
train_mask[train_idx] = True
val_mask[val_idx]     = True
test_mask[test_idx]   = True

data.train_mask = train_mask
data.val_mask   = val_mask
data.test_mask  = test_mask

# Print class distribution per split
print("\n── Stratified Split Class Distribution ──────────────────")
print(f"  Total  : {data.num_nodes:,}  |  Train: {train_mask.sum().item():,}  "
      f"Val: {val_mask.sum().item():,}  Test: {test_mask.sum().item():,}")
print(f"\n  {'Class':>5} | {'Train':>6} {'(%)':>6} | {'Val':>5} {'(%)':>6} | {'Test':>5} {'(%)':>6}")
print(f"  {'─'*5} | {'─'*6} {'─'*6} | {'─'*5} {'─'*6} | {'─'*5} {'─'*6}")
for c in range(dataset.num_classes):
    tr = (labels[train_idx] == c).sum()
    va = (labels[val_idx]   == c).sum()
    te = (labels[test_idx]  == c).sum()
    print(f"  {c:>5} | {tr:>6} {tr/len(train_idx)*100:>5.1f}% | "
          f"{va:>5} {va/len(val_idx)*100:>5.1f}% | "
          f"{te:>5} {te/len(test_idx)*100:>5.1f}%")

# Move data to device
data = data.to(device)

# ─────────────────────────────────────────────────────────────
# SHARED EVALUATE FUNCTION
# ─────────────────────────────────────────────────────────────
def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        out  = model(data.x, data.edge_index)
        pred = out[mask].argmax(dim=1)
        labels_m = data.y[mask]
        acc  = (pred == labels_m).float().mean().item()
        f1   = f1_score(labels_m.cpu(), pred.cpu(),
                        average='macro', zero_division=0)
    return acc, f1

# ─────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────
from torch_geometric.nn import GATConv, SAGEConv, GINConv
from torch.nn import Linear, BatchNorm1d, ReLU, Sequential

# ── MLP ───────────────────────────────────────────────────────
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = Linear(767, 256)
        self.drop = torch.nn.Dropout(0.5)
        self.fc2  = Linear(256, 10)
    def forward(self, x, edge_index=None):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

# ── GAT ───────────────────────────────────────────────────────
class GAT(torch.nn.Module):
    def __init__(self, dropout, hidden_dim):
        super().__init__()
        per_head = hidden_dim // 8
        self.conv1 = GATConv(767, per_head, heads=8,
                             dropout=dropout, concat=True)
        self.conv2 = GATConv(per_head * 8, 10, heads=1,
                             dropout=dropout, concat=False)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

# ── GraphSAGE ─────────────────────────────────────────────────
class GraphSAGE(torch.nn.Module):
    def __init__(self, dropout, hidden_dim):
        super().__init__()
        self.conv1   = SAGEConv(767, hidden_dim, aggr='mean')
        self.conv2   = SAGEConv(hidden_dim, 10)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

# ── GIN ───────────────────────────────────────────────────────
class GIN(torch.nn.Module):
    def __init__(self, dropout, hidden_dim):
        super().__init__()
        mlp1 = Sequential(
            Linear(767, hidden_dim), BatchNorm1d(hidden_dim),
            ReLU(), Linear(hidden_dim, hidden_dim))
        mlp2 = Sequential(
            Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim),
            ReLU(), Linear(hidden_dim, 10))
        self.conv1   = GINConv(mlp1, train_eps=True)
        self.conv2   = GINConv(mlp2, train_eps=True)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

# ─────────────────────────────────────────────────────────────
# HYPERPARAMETER GRID SEARCH (50 epochs per combo)
# ─────────────────────────────────────────────────────────────
GRID = {
    'learning_rate': [0.001, 0.01],
    'dropout':       [0.3,   0.5 ],
    'hidden_dim':    [128,   256 ],
}

def grid_combos():
    combos = []
    for lr in GRID['learning_rate']:
        for d in GRID['dropout']:
            for h in GRID['hidden_dim']:
                combos.append({'lr': lr, 'dropout': d, 'hidden': h})
    return combos   # 8 combos

def run_grid_search(model_cls, model_label):
    print(f"\n── Grid Search: {model_label} ({'─'*(40-len(model_label))})")
    combos  = grid_combos()
    results = []
    for cfg in combos:
        torch.manual_seed(42)
        model = model_cls(cfg['dropout'], cfg['hidden']).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=cfg['lr'],
                                 weight_decay=5e-4)
        best_va = 0.0
        for epoch in range(50):
            model.train()
            opt.zero_grad()
            out  = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            opt.step()
            va, _ = evaluate(model, data, data.val_mask)
            if va > best_va:
                best_va = va
        results.append({**cfg, 'val_acc': best_va})
        print(f"  lr={cfg['lr']:.3f} | drop={cfg['dropout']} | "
              f"hidden={cfg['hidden']:3d} → val_acc={best_va:.4f}")

    results.sort(key=lambda x: x['val_acc'], reverse=True)
    print(f"\n  Top-3 configs for {model_label}:")
    print(f"  {'Rank':>4} | {'lr':>6} | {'dropout':>7} | {'hidden':>6} | {'Val Acc (50ep)':>14}")
    print(f"  {'─'*4} | {'─'*6} | {'─'*7} | {'─'*6} | {'─'*14}")
    for i, r in enumerate(results[:3], 1):
        print(f"  {i:>4} | {r['lr']:>6} | {r['dropout']:>7} | "
              f"{r['hidden']:>6} | {r['val_acc']:>14.4f}")
    best = results[0]
    print(f"\n  ✓ Best config for {model_label}: "
          f"lr={best['lr']}, dropout={best['dropout']}, hidden={best['hidden']}")
    return best

# ─────────────────────────────────────────────────────────────
# FULL TRAINING FUNCTION (200 epochs)
# ─────────────────────────────────────────────────────────────
def full_train(model_cls, cfg, checkpoint_path, model_label,
               fixed_lr=None, fixed_dropout=None, fixed_hidden=None):
    """Train for 200 epochs with best config. Returns metrics + curves."""
    lr      = fixed_lr      if fixed_lr      is not None else cfg['lr']
    dropout = fixed_dropout if fixed_dropout is not None else cfg['dropout']
    hidden  = fixed_hidden  if fixed_hidden  is not None else cfg['hidden']

    torch.manual_seed(42)
    if model_cls.__name__ == 'MLP':
        model = model_cls().to(device)
    else:
        model = model_cls(dropout, hidden).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    train_losses, val_accs = [], []
    best_val, best_val_ep  = 0.0, 0

    start = time.time()
    for epoch in range(200):
        model.train()
        opt.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        opt.step()

        va, _ = evaluate(model, data, data.val_mask)
        train_losses.append(loss.item())
        val_accs.append(va)

        if va > best_val:
            best_val = va
            best_val_ep = epoch
            torch.save(model.state_dict(), checkpoint_path)

        if (epoch + 1) % 50 == 0:
            print(f"  [{model_label}] Epoch {epoch+1:3d}/200 | "
                  f"loss={loss.item():.4f} | val_acc={va:.4f} | "
                  f"best={best_val:.4f} (ep {best_val_ep+1})")

    elapsed = time.time() - start

    # Load best checkpoint & evaluate test
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_acc, test_f1 = evaluate(model, data, data.test_mask)

    print(f"\n  ✓ {model_label} done | best_val={best_val:.4f} | "
          f"test_acc={test_acc:.4f} | macro_f1={test_f1:.4f} | "
          f"time={elapsed:.1f}s")

    return {
        'best_val':    best_val,
        'test_acc':    test_acc,
        'f1':          test_f1,
        'time':        elapsed,
        'train_losses': train_losses,
        'val_accs':    val_accs,
    }

# ─────────────────────────────────────────────────────────────
# MLP BASELINE (no graph structure)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("MLP BASELINE (no graph structure — feature lower bound)")
print("="*65)
mlp_res = full_train(MLP, cfg=None, checkpoint_path='amazon_mlp_best.pt',
                     model_label='MLP', fixed_lr=0.01,
                     fixed_dropout=0.5, fixed_hidden=256)

# ─────────────────────────────────────────────────────────────
# GAT
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("MODEL 1: GAT")
print("="*65)
gat_best = run_grid_search(GAT, 'GAT')
print(f"\n── Full Training: GAT (200 epochs, winning config) ──────")
gat_res  = full_train(GAT, gat_best, 'amazon_gat_best.pt', 'GAT')

# ─────────────────────────────────────────────────────────────
# GraphSAGE
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("MODEL 2: GraphSAGE")
print("="*65)
sage_best = run_grid_search(GraphSAGE, 'GraphSAGE')
print(f"\n── Full Training: GraphSAGE (200 epochs, winning config) ─")
sage_res  = full_train(GraphSAGE, sage_best, 'amazon_sage_best.pt', 'GraphSAGE')

# ─────────────────────────────────────────────────────────────
# GIN
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("MODEL 3: GIN")
print("="*65)
gin_best = run_grid_search(GIN, 'GIN')
print(f"\n── Full Training: GIN (200 epochs, winning config) ──────")
gin_res  = full_train(GIN, gin_best, 'amazon_gin_best.pt', 'GIN')

# ─────────────────────────────────────────────────────────────
# TRAINING CURVES — 4 subplots
# ─────────────────────────────────────────────────────────────
print("\n── Saving training curves plot ──────────────────────────")
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle("Amazon Computers — training curves (tuned configs)",
             fontsize=14, fontweight='bold', y=1.02)

configs = [
    ('MLP (no graph)', mlp_res,  'lr=0.01/drop=0.5/h=256'),
    ('GAT',            gat_res,  f"lr={gat_best['lr']}/drop={gat_best['dropout']}/h={gat_best['hidden']}"),
    ('GraphSAGE',      sage_res, f"lr={sage_best['lr']}/drop={sage_best['dropout']}/h={sage_best['hidden']}"),
    ('GIN',            gin_res,  f"lr={gin_best['lr']}/drop={gin_best['dropout']}/h={gin_best['hidden']}"),
]

for ax, (title, res, cfg_str) in zip(axes, configs):
    epochs = range(1, 201)
    color1, color2 = '#2563EB', '#F97316'
    ax2 = ax.twinx()
    ax.plot(epochs,  res['train_losses'], color=color1, linewidth=1.5, label='Train Loss')
    ax2.plot(epochs, res['val_accs'],     color=color2, linewidth=1.5, label='Val Acc')
    ax.set_title(f"{title}\n{cfg_str}", fontsize=9)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Loss', color=color1)
    ax2.set_ylabel('Val Acc',   color=color2)
    ax.tick_params(axis='y', labelcolor=color1)
    ax2.tick_params(axis='y', labelcolor=color2)
    lines  = ax.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig('amazon_all_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved → amazon_all_curves.png")

# ─────────────────────────────────────────────────────────────
# MASTER COMPARISON TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("MILESTONE 2 — MASTER COMPARISON TABLE")
print("="*65)

rows = [
    ('MLP (no graph)', '—',
     f"lr=0.01/d=0.5/h=256",
     mlp_res),
    ('GAT',
     f"lr={gat_best['lr']}/d={gat_best['dropout']}/h={gat_best['hidden']}",
     f"lr={gat_best['lr']}/d={gat_best['dropout']}/h={gat_best['hidden']}",
     gat_res),
    ('GraphSAGE',
     f"lr={sage_best['lr']}/d={sage_best['dropout']}/h={sage_best['hidden']}",
     f"lr={sage_best['lr']}/d={sage_best['dropout']}/h={sage_best['hidden']}",
     sage_res),
    ('GIN',
     f"lr={gin_best['lr']}/d={gin_best['dropout']}/h={gin_best['hidden']}",
     f"lr={gin_best['lr']}/d={gin_best['dropout']}/h={gin_best['hidden']}",
     gin_res),
]

hdr = (f"{'Model':<18} | {'Best Config':^28} | {'Best Val Acc':>12} | "
       f"{'Test Acc':>9} | {'Macro F1':>9} | {'Time(s)':>8}")
print(hdr)
print("─" * len(hdr))
for model_name, cfg_str, _, res in rows:
    print(f"{model_name:<18} | {cfg_str:^28} | "
          f"{res['best_val']:>12.4f} | {res['test_acc']:>9.4f} | "
          f"{res['f1']:>9.4f} | {res['time']:>8.1f}")

# ─────────────────────────────────────────────────────────────
# SAVE TO results.json
# ─────────────────────────────────────────────────────────────
print("\n── Saving results.json ──────────────────────────────────")

save_results('amazon', 'MLP', {
    'best_cfg':  '—',
    'val_acc':   round(mlp_res['best_val'], 4),
    'test_acc':  round(mlp_res['test_acc'],  4),
    'f1':        round(mlp_res['f1'],        4),
    'time':      round(mlp_res['time'],      2),
})
save_results('amazon', 'GAT', {
    'best_cfg':  f"lr={gat_best['lr']},d={gat_best['dropout']},h={gat_best['hidden']}",
    'val_acc':   round(gat_res['best_val'], 4),
    'test_acc':  round(gat_res['test_acc'],  4),
    'f1':        round(gat_res['f1'],        4),
    'time':      round(gat_res['time'],      2),
})
save_results('amazon', 'SAGE', {
    'best_cfg':  f"lr={sage_best['lr']},d={sage_best['dropout']},h={sage_best['hidden']}",
    'val_acc':   round(sage_res['best_val'], 4),
    'test_acc':  round(sage_res['test_acc'],  4),
    'f1':        round(sage_res['f1'],        4),
    'time':      round(sage_res['time'],      2),
})
save_results('amazon', 'GIN', {
    'best_cfg':  f"lr={gin_best['lr']},d={gin_best['dropout']},h={gin_best['hidden']}",
    'val_acc':   round(gin_res['best_val'], 4),
    'test_acc':  round(gin_res['test_acc'],  4),
    'f1':        round(gin_res['f1'],        4),
    'time':      round(gin_res['time'],      2),
})

print("  Saved → results.json")
print("\n" + "="*65)
print("MILESTONE 2 COMPLETE")
print("="*65)