# --- SEED BLOCK (paste at the very top) ---
import torch, numpy as np, random, time, json, os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import (cohen_kappa_score, f1_score,
                             precision_score, recall_score)
torch.manual_seed(42); np.random.seed(42); random.seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def save_results(dataset_key, model_name, metrics):
    path = 'results.json'
    data = json.load(open(path)) if os.path.exists(path) else {}
    if dataset_key not in data: data[dataset_key] = {}
    data[dataset_key][model_name] = metrics
    json.dump(data, open(path,'w'), indent=2)

# ---------------------- MILestone 2: Airports ----------------------
import itertools
from collections import defaultdict
from torch import nn
from torch_geometric.datasets import Airports
from torch_geometric.nn import GATConv, SAGEConv, GINConv
from torch_geometric.utils import to_undirected, degree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

# Load dataset
dataset = Airports(root='.', name='USA')
data = dataset[0]
num_nodes = data.num_nodes  # 1190
edge_index = data.edge_index

# Load splits
splits = torch.load('splits_usa.pt', map_location='cpu')
train_idx = splits['train']
val_idx = splits['val']
test_idx = splits['test']
labels = data.y.view(-1).to(torch.long)

print("Splits:", {k: int(v.numel()) for k,v in splits.items()})

# --- Shared evaluation exactly as specified ---
def evaluate(model, embed, data, idx, formulation, device):
    model.eval(); embed.eval()
    with torch.no_grad():
        x = embed(torch.arange(data.num_nodes).to(device))
        labels = data.y[idx].to(device)
        if formulation=='ordinal':
            raw  = model(x,data.edge_index.to(device)).squeeze()[idx]
            pred = raw.clamp(0,3).round().long()
            mse  = F.mse_loss(raw,labels.float()).item()
            mae  = (raw-labels.float()).abs().mean().item()
        else:
            logits = model(x,data.edge_index.to(device))[idx]
            pred   = logits.argmax(dim=1)
            mse    = F.mse_loss(pred.float(),labels.float()).item()
            mae    = (pred.float()-labels.float()).abs().mean().item()
        lnp = labels.cpu().numpy(); pnp = pred.cpu().numpy()
        acc   = (pnp==lnp).mean()
        kappa = cohen_kappa_score(lnp,pnp,weights='quadratic')
        f1    = f1_score(lnp,pnp,average='macro',zero_division=0)
    return {'mse':mse,'mae':mae,'acc':acc,'kappa':kappa,'f1':f1}

# ---------------------- Degree heuristic baseline ----------------------
def degree_heuristic_baseline(data, train_idx, test_idx, labels, k=10, eps=1e-6):
    # Compute undirected degrees for a clean neighbor-degree notion
    edge_index_u = to_undirected(data.edge_index)
    deg = degree(edge_index_u[0], num_nodes=data.num_nodes).float()

    deg_train = deg[train_idx]                  # [n_train]
    y_train = labels[train_idx].float()        # [n_train]
    deg_test = deg[test_idx]                    # [n_test]
    y_test = labels[test_idx]                  # [n_test]

    preds = []
    for dt in deg_test:
        dist = (deg_train - dt).abs()          # [n_train]
        topk = torch.topk(dist, k=min(k, dist.numel()), largest=False).indices
        dsel = dist[topk]
        w = 1.0 / (dsel + eps)
        wmean = (w * y_train[topk]).sum() / w.sum()
        p = int(torch.round(wmean).item())
        p = max(0, min(3, p))
        preds.append(p)
    preds = torch.tensor(preds, dtype=torch.long)

    lnp = y_test.cpu().numpy()
    pnp = preds.cpu().numpy()
    acc = (pnp == lnp).mean()
    kappa = cohen_kappa_score(lnp, pnp, weights='quadratic')
    f1 = f1_score(lnp, pnp, average='macro', zero_division=0)
    mse = F.mse_loss(preds.float(), y_test.float()).item()
    mae = (preds.float() - y_test.float()).abs().mean().item()
    return {'mse': mse, 'mae': mae, 'acc': acc, 'kappa': kappa, 'f1': f1}

degree_heur = degree_heuristic_baseline(data, train_idx, test_idx, labels, k=10)
save_results('airports', 'degree_heuristic', degree_heur)
print("Degree heuristic test:", degree_heur)

# ---------------------- Model bodies (conv identical across formulations) ----------------------
class GATNet(nn.Module):
    def __init__(self, embed_dim, dropout, formulation):
        super().__init__()
        self.gat1 = GATConv(embed_dim, embed_dim // 8, heads=8, dropout=dropout, concat=True)
        self.gat2 = GATConv(embed_dim, 32, heads=1, concat=False)
        out_dim = 1 if formulation == 'ordinal' else 4
        self.head = nn.Linear(32, out_dim)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        return self.head(x)

class SAGENet(nn.Module):
    def __init__(self, embed_dim, dropout, formulation):
        super().__init__()
        self.dropout_p = dropout
        self.sage1 = SAGEConv(embed_dim, embed_dim, aggr='mean')
        self.sage2 = SAGEConv(embed_dim, 32)
        out_dim = 1 if formulation == 'ordinal' else 4
        self.head = nn.Linear(32, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.sage1(x, edge_index))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.sage2(x, edge_index))
        return self.head(x)

class GINNet(nn.Module):
    def __init__(self, embed_dim, dropout, formulation):
        super().__init__()
        self.dropout_p = dropout

        mlp1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.gin1 = GINConv(mlp1, train_eps=True)

        mlp2 = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        self.gin2 = GINConv(mlp2, train_eps=True)

        out_dim = 1 if formulation == 'ordinal' else 4
        self.head = nn.Linear(32, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.gin1(x, edge_index))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.gin2(x, edge_index))
        return self.head(x)

def build_model(arch, embed_dim, dropout, formulation):
    # Input: nn.Embedding(1190, embed_dim) for ALL models
    embed = nn.Embedding(num_nodes, embed_dim)
    if arch == 'GAT':
        model = GATNet(embed_dim, dropout, formulation)
    elif arch == 'GraphSAGE':
        model = SAGENet(embed_dim, dropout, formulation)
    elif arch == 'GIN':
        model = GINNet(embed_dim, dropout, formulation)
    else:
        raise ValueError(arch)
    return model.to(device), embed.to(device)

# ---------------------- Hyperparameter tuning ----------------------
grid = list(itertools.product([0.001, 0.005], [0.0, 0.3], [32, 64]))  # (lr, dropout, embed_dim)
# Grid: 8 combinations per model
archs = ['GAT', 'GraphSAGE', 'GIN']

best_cfg_by_arch = {}
tune_val_mse_by_arch = defaultdict(list)

for arch in archs:
    print("\n=== Tuning", arch, "===")
    results = []  # (val_mse, lr, dropout, embed_dim)

    node_ids = torch.arange(num_nodes, device=device)
    edge_index_d = edge_index.to(device)
    labels_d = labels.to(device)
    train_labels = labels_d[train_idx]
    val_labels = labels_d[val_idx]

    for (lr, d, e) in grid:
        torch.manual_seed(42)
        model, embed = build_model(arch, e, d, formulation='ordinal')
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        opt_embed = torch.optim.Adam(embed.parameters(), lr=lr, weight_decay=5e-4)

        # Train for 75 epochs (seed=42, same splits, MSELoss ordinal)
        for epoch in range(75):
            model.train(); embed.train()
            opt.zero_grad(set_to_none=True)
            opt_embed.zero_grad(set_to_none=True)

            x = embed(node_ids)
            raw = model(x, edge_index_d).squeeze()  # [num_nodes]
            loss = F.mse_loss(raw[train_idx], train_labels.float())
            loss.backward()
            opt.step()
            opt_embed.step()

        # Val MSE for this config
        model.eval(); embed.eval()
        with torch.no_grad():
            x = embed(node_ids)
            raw = model(x, edge_index_d).squeeze()
            val_mse = F.mse_loss(raw[val_idx], val_labels.float()).item()

        results.append((val_mse, lr, d, e))
        tune_val_mse_by_arch[arch].append((lr, d, e, val_mse))
        print(f"cfg: lr={lr}, dropout={d}, embed_dim={e} -> val_MSE={val_mse:.6f}")

    results.sort(key=lambda t: t[0])
    top3 = results[:3]
    print("Top-3 configs for", arch)
    print("| Rank | lr | dropout | embed_dim | Val MSE (75 ep) |")
    for rank, (vmse, lr, d, e) in enumerate(top3, start=1):
        print(f"| {rank} | {lr} | {d} | {e} | {vmse:.6f} |")

    best_val_mse, best_lr, best_d, best_e = results[0]
    best_cfg_by_arch[arch] = {'lr': best_lr, 'dropout': best_d, 'embed_dim': best_e}
    print(f"Best config for {arch}: lr={best_lr}, dropout={best_d}, embed={best_e}")

# ---------------------- Full training (300 epochs) for BOTH formulations ----------------------
def train_one(arch, formulation, lr, dropout, embed_dim):
    # Checkpoint naming:
    # airports_{gat,sage,gin}_{ordinal,cls}_best.pt
    arch_key = arch.lower()
    if arch_key == 'graphsage':
        arch_key = 'sage'
    ckpt_name = f"airports_{arch_key}_{'ordinal' if formulation=='ordinal' else 'cls'}_best.pt"

    seeds = [42, 123, 456, 789, 999]
    all_metrics = []
    first_train_mse_hist = []
    first_val_mse_hist = []

    for i, current_seed in enumerate(seeds):
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        random.seed(current_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(current_seed)

        model, embed = build_model(arch, embed_dim, dropout, formulation=formulation)
        node_ids = torch.arange(num_nodes, device=device)
        edge_index_d = edge_index.to(device)
        labels_d = labels.to(device)

        opt = torch.optim.Adam(list(model.parameters()) + list(embed.parameters()),
                               lr=lr, weight_decay=5e-4)

        # For curves
        train_mse_hist = []
        val_mse_hist = []

        best_path = ckpt_name.replace('.pt', f'_{current_seed}.pt')
        if formulation == 'ordinal':
            best_val_mse = float('inf')
        else:
            best_val_acc = -float('inf')

        start = time.time()

        # Precompute label tensors for speed
        y_train = labels_d[train_idx]
        y_val = labels_d[val_idx]

        for epoch in range(300):
            model.train(); embed.train()
            opt.zero_grad(set_to_none=True)

            x = embed(node_ids)
            out = model(x, edge_index_d)

            if formulation == 'ordinal':
                raw = out.squeeze()  # [num_nodes]
                loss = F.mse_loss(raw[train_idx], y_train.float())
            else:
                logits = out  # [num_nodes, 4]
                loss = F.cross_entropy(logits[train_idx], y_train)

            loss.backward()
            opt.step()

            # Train/Val MSE for plotting (match evaluate's mse definitions)
            model.eval(); embed.eval()
            with torch.no_grad():
                x = embed(node_ids)
                out = model(x, edge_index_d)

                if formulation == 'ordinal':
                    raw = out.squeeze()
                    train_mse = F.mse_loss(raw[train_idx], y_train.float()).item()
                    val_mse = F.mse_loss(raw[val_idx], y_val.float()).item()
                else:
                    logits = out
                    train_pred = logits[train_idx].argmax(dim=1)
                    val_pred = logits[val_idx].argmax(dim=1)
                    train_mse = F.mse_loss(train_pred.float(), y_train.float()).item()
                    val_mse = F.mse_loss(val_pred.float(), y_val.float()).item()

            train_mse_hist.append(train_mse)
            val_mse_hist.append(val_mse)

            # Checkpoint selection
            if formulation == 'ordinal':
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    torch.save({'embed': embed.state_dict(), 'model': model.state_dict()}, best_path)
            else:
                with torch.no_grad():
                    x = embed(node_ids)
                    logits = model(x, edge_index_d)
                    val_pred = logits[val_idx].argmax(dim=1)
                    val_acc = (val_pred == y_val).float().mean().item()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save({'embed': embed.state_dict(), 'model': model.state_dict()}, best_path)

        elapsed = time.time() - start

        # Load best checkpoint
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        embed.load_state_dict(ckpt['embed'])

        # Test evaluation
        test_metrics = evaluate(model, embed, data, test_idx, formulation=formulation, device=device)
        test_metrics['time'] = elapsed
        all_metrics.append(test_metrics)

        if i == 0:
            first_train_mse_hist = train_mse_hist
            first_val_mse_hist = val_mse_hist

    keys = ['mse', 'mae', 'acc', 'kappa', 'f1', 'time']
    agg_metrics = {}
    for k in keys:
        vals = [m[k] for m in all_metrics]
        agg_metrics[k] = float(np.mean(vals))
        agg_metrics[f"{k}_std"] = float(np.std(vals))

    print(f"  ✓ {arch} {formulation} done (5 seeds) | time={agg_metrics['time']:.1f}s")
    
    return {
        'ckpt_path': ckpt_name,
        'train_mse_hist': first_train_mse_hist,
        'val_mse_hist': first_val_mse_hist,
        **agg_metrics
    }

# Train all 6 pairs and build master table
master_rows = []

# For plots
history_ordinal = {}
history_cls = {}

for arch in archs:
    cfg = best_cfg_by_arch[arch]
    lr, dropout, embed_dim = cfg['lr'], cfg['dropout'], cfg['embed_dim']

    print(f"\n--- Training {arch} Ordinal (300ep) ---")
    ord_res = train_one(arch, 'ordinal', lr=lr, dropout=dropout, embed_dim=embed_dim)
    save_results('airports', f"{arch if arch!='GraphSAGE' else 'GraphSAGE'}_ordinal",
                 {'best_cfg': cfg, 
                  'mse': ord_res['mse'], 'mse_std': ord_res['mse_std'],
                  'mae': ord_res['mae'], 'mae_std': ord_res['mae_std'],
                  'acc': ord_res['acc'], 'acc_std': ord_res['acc_std'],
                  'kappa': ord_res['kappa'], 'kappa_std': ord_res['kappa_std'],
                  'f1': ord_res['f1'], 'f1_std': ord_res['f1_std'],
                  'time': ord_res['time'], 'time_std': ord_res['time_std']})
    master_rows.append({
        'Model+Formulation': f"{arch} — Ordinal",
        'Best Config': f"lr={lr}, d={dropout}, embed={embed_dim}",
        **{k: f"{ord_res[k]:.4f}±{ord_res[k+'_std']:.4f}" for k in ['mse', 'mae', 'acc', 'kappa', 'f1']},
        'Time': f"{ord_res['time']:.1f}±{ord_res['time_std']:.1f}"
    })
    history_ordinal[arch] = (ord_res['train_mse_hist'], ord_res['val_mse_hist'])

    print(f"\n--- Training {arch} Classification (300ep) ---")
    cls_res = train_one(arch, 'cls', lr=lr, dropout=dropout, embed_dim=embed_dim)
    save_results('airports', f"{arch if arch!='GraphSAGE' else 'GraphSAGE'}_cls",
                 {'best_cfg': cfg, 
                  'mse': cls_res['mse'], 'mse_std': cls_res['mse_std'],
                  'mae': cls_res['mae'], 'mae_std': cls_res['mae_std'],
                  'acc': cls_res['acc'], 'acc_std': cls_res['acc_std'],
                  'kappa': cls_res['kappa'], 'kappa_std': cls_res['kappa_std'],
                  'f1': cls_res['f1'], 'f1_std': cls_res['f1_std'],
                  'time': cls_res['time'], 'time_std': cls_res['time_std']})
    master_rows.append({
        'Model+Formulation': f"{arch} — Classification",
        'Best Config': f"lr={lr}, d={dropout}, embed={embed_dim}",
        **{k: f"{cls_res[k]:.4f}±{cls_res[k+'_std']:.4f}" for k in ['mse', 'mae', 'acc', 'kappa', 'f1']},
        'Time': f"{cls_res['time']:.1f}±{cls_res['time_std']:.1f}"
    })
    history_cls[arch] = (cls_res['train_mse_hist'], cls_res['val_mse_hist'])

# Add degree heuristic to master rows first
master_table = []
master_table.append({
    'Model+Formulation': "Degree heuristic",
    'Best Config': "—",
    'mse': f"{degree_heur['mse']:.4f}",
    'mae': f"{degree_heur['mae']:.4f}",
    'acc': f"{degree_heur['acc']:.4f}",
    'kappa': f"{degree_heur['kappa']:.4f}",
    'f1': f"{degree_heur['f1']:.4f}",
    'Time': "—"
})
master_table.extend(master_rows)

print("\n=== MASTER TABLE ===")
print(f"| {'Model+Formulation':<29} | {'Best Config':<23} | {'MSE':>15} | {'MAE':>15} | {'Acc':>15} | {'Kappa':>15} | {'F1':>15} | {'Time':>13} |")
print("|-------------------------------|-------------------------|-----------------|-----------------|-----------------|-----------------|-----------------|---------------|")
for r in master_table:
    print(f"| {r['Model+Formulation']:<29} | {r['Best Config']:<23} | "
          f"{r['mse']:>15} | {r['mae']:>15} | {r['acc']:>15} | {r['kappa']:>15} | {r['f1']:>15} | {r['Time']:>13} |")

# ---------------------- FIGURES ----------------------
def plot_curves(hist_dict, out_name, title_prefix):
    # hist_dict: arch -> (train_mse_hist, val_mse_hist)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    order = ['GAT', 'GraphSAGE', 'GIN']
    for ax, arch in zip(axes, order):
        train_hist, val_hist = hist_dict[arch]
        epochs = list(range(1, len(train_hist) + 1))

        ax.plot(epochs, train_hist, color='blue', label='train MSE')
        ax.set_xlabel('epoch')
        ax.set_ylabel('train MSE', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_title(f"{arch}")

        ax2 = ax.twinx()
        ax2.plot(epochs, val_hist, color='orange', label='val MSE')
        ax2.set_ylabel('val MSE', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        ax.grid(True, alpha=0.25)

    fig.suptitle(title_prefix)
    fig.tight_layout()
    plt.savefig(out_name, dpi=150)
    plt.close(fig)

plot_curves(history_ordinal, 'airports_ordinal_curves.png', "airports_ordinal_curves.png")
plot_curves(history_cls, 'airports_cls_curves.png', "airports_cls_curves.png")
print("Saved figures: airports_ordinal_curves.png, airports_cls_curves.png")

print("\nDone. Key outputs:")
print("- results.json")
print("- airports_{gat|sage|gin}_{ordinal|cls}_best.pt")
print("- airports_ordinal_curves.png, airports_cls_curves.png")