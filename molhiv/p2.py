import torch, numpy as np, random, time, json, os, sys
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, GATv2Conv, SAGEConv, global_mean_pool
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

torch.manual_seed(42); np.random.seed(42); random.seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torch_geometric
print(f"Python version: {sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")
print(f"torch_geometric version: {torch_geometric.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

def save_results(dataset_key, model_name, metrics):
    path = 'results.json'
    data = json.load(open(path)) if os.path.exists(path) else {}
    if dataset_key not in data: data[dataset_key] = {}
    data[dataset_key][model_name] = metrics
    json.dump(data, open(path,'w'), indent=2)

print("\nPROJECT REQUIREMENT — MILESTONE 2:")
print("All 3 GNN architectures trained on OGBG-MOLHIV with hyperparameter tuning.")
print("GraphSAGE intentionally receives no bond edge features — structural-only control.\n")

print("# ARCHITECTURE RATIONALE (Milestone 2 — Model Selection):")
print("# GAT  — attention-weighted aggregation. Subsumes GCN and adds selective neighbour weighting.")
print("# GraphSAGE — inductive mean aggregation; explicit no-edge-feature control on MOLHIV.")
print("# GIN  — sum aggregator; most expressive per Weisfeiler-Lehman test.")

# Dataset re-load to ensure clean state
dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='.')
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbg-molhiv')

subset_train = dataset[split_idx['train']]
labels = subset_train.y.view(-1)
pos_train = int((labels == 1).sum().item())
neg_train = len(labels) - pos_train
pos_weight = neg_train / pos_train

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float))

class AtomEncoder(nn.Module):
    # Added buffer to max values just to be absolutely safe with OGB embeddings
    VOCAB=[119,5,10,6,8,2,6,2,2]
    def __init__(self,d=300):
        super().__init__()
        self.embs=nn.ModuleList([nn.Embedding(v + 10, d) for v in self.VOCAB])
    def forward(self,x): 
        return sum(e(x[:,i]) for i,e in enumerate(self.embs))

class BondEncoder(nn.Module):
    VOCAB=[5,6,2]
    def __init__(self,d=300):
        super().__init__()
        self.embs=nn.ModuleList([nn.Embedding(v + 10, d) for v in self.VOCAB])
    def forward(self,ea): 
        return sum(e(ea[:,i]) for i,e in enumerate(self.embs))

def evaluate(model, loader, evaluator, device):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            out = model(b).squeeze(-1)
            yp.append(out.cpu())
            yt.append(b.y.squeeze(-1).cpu())
    yt = torch.cat(yt).float()
    yp = torch.cat(yp).float()
    # OGB Evaluator expects shape (num_samples, 1) and specific dict keys
    roc = evaluator.eval({'y_true': yt.unsqueeze(1), 'y_pred': yp.unsqueeze(1)})['rocauc']
    pr = torch.sigmoid(yp)
    pd = (pr > 0.5).long()
    return {
        'roc_auc': roc,
        'precision': precision_score(yt, pd, zero_division=0),
        'recall': recall_score(yt, pd, zero_division=0),
        'f1': f1_score(yt, pd, zero_division=0)
    }

class GIN(nn.Module):
    def __init__(self, num_layers, dropout=0.5):
        super().__init__()
        self.atom_enc = AtomEncoder(300)
        self.bond_enc = BondEncoder(300)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        self.virtualnode_emb = nn.Embedding(1, 300)
        self.mlp_virtualnode_list = nn.ModuleList([
            nn.Sequential(nn.Linear(300, 300), nn.BatchNorm1d(300), nn.ReLU(), nn.Linear(300, 300), nn.BatchNorm1d(300), nn.ReLU()) 
            for _ in range(num_layers - 1)
        ])
        for _ in range(num_layers):
            mlp = nn.Sequential(nn.Linear(300,300), nn.BatchNorm1d(300), nn.ReLU(), nn.Linear(300,300))
            self.convs.append(GINEConv(mlp, edge_dim=300))
            self.bns.append(nn.BatchNorm1d(300))
        self.pool = global_mean_pool
        self.lin = nn.Linear(300,1)

    def forward(self, b):
        x = self.atom_enc(b.x)
        ea = self.bond_enc(b.edge_attr)
        virtualnode_embedding = self.virtualnode_emb(torch.zeros(b.batch[-1].item() + 1).long().to(x.device))
        
        for layer in range(len(self.convs)):
            x = x + virtualnode_embedding[b.batch]
            
            h = self.convs[layer](x, b.edge_index, ea)
            h = self.bns[layer](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = h + x
            
            if layer < len(self.convs) - 1:
                virtualnode_embedding = virtualnode_embedding + global_mean_pool(x, b.batch)
                virtualnode_embedding = self.mlp_virtualnode_list[layer](virtualnode_embedding)
                virtualnode_embedding = F.dropout(virtualnode_embedding, p=self.dropout, training=self.training)
                
        x = self.pool(x, b.batch)
        return self.lin(x)

class GAT(nn.Module):
    def __init__(self, num_layers, dropout=0.5):
        super().__init__()
        self.atom_enc = AtomEncoder(300)
        self.bond_enc = BondEncoder(300)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        self.virtualnode_emb = nn.Embedding(1, 300)
        self.mlp_virtualnode_list = nn.ModuleList([
            nn.Sequential(nn.Linear(300, 300), nn.BatchNorm1d(300), nn.ReLU(), nn.Linear(300, 300), nn.BatchNorm1d(300), nn.ReLU()) 
            for _ in range(num_layers - 1)
        ])
        for _ in range(num_layers):
            self.convs.append(GATv2Conv(300, 75, heads=4, concat=True, edge_dim=300))
            self.bns.append(nn.BatchNorm1d(300))
        self.pool = global_mean_pool
        self.lin = nn.Linear(300,1)

    def forward(self, b):
        x = self.atom_enc(b.x)
        ea = self.bond_enc(b.edge_attr)
        virtualnode_embedding = self.virtualnode_emb(torch.zeros(b.batch[-1].item() + 1).long().to(x.device))
        
        for layer in range(len(self.convs)):
            x = x + virtualnode_embedding[b.batch]
            
            h = self.convs[layer](x, b.edge_index, ea)
            h = self.bns[layer](h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = h + x
            
            if layer < len(self.convs) - 1:
                virtualnode_embedding = virtualnode_embedding + global_mean_pool(x, b.batch)
                virtualnode_embedding = self.mlp_virtualnode_list[layer](virtualnode_embedding)
                virtualnode_embedding = F.dropout(virtualnode_embedding, p=self.dropout, training=self.training)
                
        x = self.pool(x, b.batch)
        return self.lin(x)

class GraphSAGE(nn.Module):
    def __init__(self, num_layers, dropout=0.5):
        super().__init__()
        self.atom_enc = AtomEncoder(300)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        self.virtualnode_emb = nn.Embedding(1, 300)
        self.mlp_virtualnode_list = nn.ModuleList([
            nn.Sequential(nn.Linear(300, 300), nn.BatchNorm1d(300), nn.ReLU(), nn.Linear(300, 300), nn.BatchNorm1d(300), nn.ReLU()) 
            for _ in range(num_layers - 1)
        ])
        for _ in range(num_layers):
            self.convs.append(SAGEConv(300, 300, aggr='mean'))
            self.bns.append(nn.BatchNorm1d(300))
        self.pool = global_mean_pool
        self.lin = nn.Linear(300,1)

    def forward(self, b):
        x = self.atom_enc(b.x)
        virtualnode_embedding = self.virtualnode_emb(torch.zeros(b.batch[-1].item() + 1).long().to(x.device))
        
        for layer in range(len(self.convs)):
            x = x + virtualnode_embedding[b.batch]
            
            h = self.convs[layer](x, b.edge_index)
            h = self.bns[layer](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = h + x
            
            if layer < len(self.convs) - 1:
                virtualnode_embedding = virtualnode_embedding + global_mean_pool(x, b.batch)
                virtualnode_embedding = self.mlp_virtualnode_list[layer](virtualnode_embedding)
                virtualnode_embedding = F.dropout(virtualnode_embedding, p=self.dropout, training=self.training)
                
        x = self.pool(x, b.batch)
        return self.lin(x)

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for b in loader:
        b = b.to(device)
        optimizer.zero_grad()
        out = model(b).squeeze(-1)
        loss = criterion.to(device)(out, b.y.squeeze(-1).float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

grid = {
    'learning_rate': [0.0001, 0.001],
    'num_layers': [3, 5],
    'batch_size': [32, 64]
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

model_classes = {'GIN': GIN, 'GAT': GAT, 'GraphSAGE': GraphSAGE}
best_configs = {}
history = {}
final_results = {}

import itertools
keys, values = zip(*grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

for name, ModelClass in model_classes.items():
    print(f"\n{'='*40}\nHyperparameter Tuning for {name}\n{'='*40}")
    results = []
    
    for i, cfg in enumerate(combinations):
        lr = cfg['learning_rate']
        n_layers = cfg['num_layers']
        bs = cfg['batch_size']
        
        torch.manual_seed(42)
        model = ModelClass(num_layers=n_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        
        train_loader = DataLoader(dataset[split_idx['train']], batch_size=bs, shuffle=True)
        val_loader = DataLoader(dataset[split_idx['valid']], batch_size=bs, shuffle=False)
        
        best_val = 0
        # 30 epochs for tuning
        for epoch in range(1, 31):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate(model, val_loader, evaluator, device)
            best_val = max(best_val, val_metrics['roc_auc'])
            
        results.append({'cfg': cfg, 'val_roc': best_val})
        print(f"Config {i+1}/8 [lr={lr}, layers={n_layers}, batch={bs}] -> Val ROC-AUC: {best_val:.4f}")
        
    results.sort(key=lambda x: x['val_roc'], reverse=True)
    print(f"\nTop-3 Configs for {name}:")
    print(f"| Rank | lr     | num_layers | batch_size | Val ROC-AUC (30 ep) |")
    for rank in range(3):
        c = results[rank]['cfg']
        print(f"| {rank+1:<4} | {c['learning_rate']:<6} | {c['num_layers']:<10} | {c['batch_size']:<10} | {results[rank]['val_roc']:.4f}              |")
    
    best_cfg = results[0]['cfg']
    best_configs[name] = best_cfg
    print(f"\n*** Best config for {name}: lr={best_cfg['learning_rate']}, layers={best_cfg['num_layers']}, batch={best_cfg['batch_size']} ***")

    # Retrain winning config for full 100 epochs (5 seeds)
    print(f"Retraining {name} with best config for 100 epochs (5 seeds)...")
    
    seeds = [42, 123, 456, 789, 999]
    all_metrics = []
    first_train_losses = []
    first_val_rocs = []
    
    train_loader = DataLoader(dataset[split_idx['train']], batch_size=best_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(dataset[split_idx['valid']], batch_size=best_cfg['batch_size'], shuffle=False)
    test_loader = DataLoader(dataset[split_idx['test']], batch_size=best_cfg['batch_size'], shuffle=False)

    for i, current_seed in enumerate(seeds):
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        random.seed(current_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(current_seed)

        model = ModelClass(num_layers=best_cfg['num_layers']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_cfg['learning_rate'], weight_decay=1e-5)
        
        best_val_roc = 0
        checkpoint_path = f'molhiv_{name.lower()}_best_{current_seed}.pt'
        
        train_losses = []
        val_rocs = []
        
        start_time = time.time()
        for epoch in range(1, 101):
            loss = train_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate(model, val_loader, evaluator, device)
            val_roc = val_metrics['roc_auc']
            
            train_losses.append(loss)
            val_rocs.append(val_roc)
            
            if val_roc > best_val_roc:
                best_val_roc = val_roc
                torch.save(model.state_dict(), checkpoint_path)
                
        train_time = time.time() - start_time
        
        if i == 0:
            first_train_losses = train_losses
            first_val_rocs = val_rocs
        
        # Load best for test evaluation
        model.load_state_dict(torch.load(checkpoint_path))
        test_metrics = evaluate(model, test_loader, evaluator, device)
        test_metrics['val_roc'] = best_val_roc
        test_metrics['time'] = train_time
        all_metrics.append(test_metrics)

    history[name] = {'loss': first_train_losses, 'val_roc': first_val_rocs}
    
    # Aggregate
    keys = ['val_roc', 'roc_auc', 'precision', 'recall', 'f1', 'time']
    agg_metrics = {}
    for k in keys:
        vals = [m[k] for m in all_metrics]
        agg_metrics[k] = float(np.mean(vals))
        agg_metrics[f"{k}_std"] = float(np.std(vals))

    final_results[name] = {
        'best_cfg': best_cfg,
        'val_roc': agg_metrics['val_roc'], 'val_roc_std': agg_metrics['val_roc_std'],
        'test_roc': agg_metrics['roc_auc'], 'test_roc_std': agg_metrics['roc_auc_std'],
        'precision': agg_metrics['precision'], 'precision_std': agg_metrics['precision_std'],
        'recall': agg_metrics['recall'], 'recall_std': agg_metrics['recall_std'],
        'f1': agg_metrics['f1'], 'f1_std': agg_metrics['f1_std'],
        'time': agg_metrics['time'], 'time_std': agg_metrics['time_std']
    }
    
    save_results('molhiv', name, final_results[name])

# ONE figure - 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Training Loss and Validation ROC-AUC vs Epoch', fontsize=16)

for ax, name in zip(axes, model_classes.keys()):
    ax1 = ax
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color=color)
    ax1.plot(history[name]['loss'], color=color, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Val ROC-AUC', color=color)
    ax2.plot(history[name]['val_roc'], color=color, label='Val ROC-AUC')
    ax2.tick_params(axis='y', labelcolor=color)
    
    ax.set_title(name)

fig.tight_layout()
plt.savefig('molhiv_all_curves.png')
plt.show()

# Print Table
print("\nFinal Results Summary:")
print(f"| {'Model':<9} | {'Best Config (lr/layers/batch)':<29} | {'Best Val ROC':>17} | {'Test ROC':>17} | {'Precision':>17} | {'Recall':>17} | {'F1':>17} | {'Time(s)':>15} |")
print("|-----------|-------------------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-----------------|")
for name, res in final_results.items():
    cfg_str = f"{res['best_cfg']['learning_rate']}/{res['best_cfg']['num_layers']}/{res['best_cfg']['batch_size']}"
    print(f"| {name:<9} | {cfg_str:<29} | {res['val_roc']:.4f}±{res['val_roc_std']:.4f} | {res['test_roc']:.4f}±{res['test_roc_std']:.4f} | {res['precision']:.4f}±{res['precision_std']:.4f} | {res['recall']:.4f}±{res['recall_std']:.4f} | {res['f1']:.4f}±{res['f1_std']:.4f} | {res['time']:.1f}±{res['time_std']:.1f} |")
