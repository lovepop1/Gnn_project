import torch, numpy as np, random, time, json, os, sys
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import (cohen_kappa_score, f1_score,
                             precision_score, recall_score)
import torch_geometric

# Patch torch.load to default weights_only=False for PyTorch 2.6+ compatibility with PyG/OGB
original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load

torch.manual_seed(42); np.random.seed(42); random.seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(f"Python version: {sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")
print(f"torch_geometric version: {torch_geometric.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

print("\nPROJECT REQUIREMENT — MILESTONE 1 + MILESTONE 3 SETUP:")
print("OGBG-MOLHIV is confirmed as the LARGEST of the three datasets (41,127 graphs")
print("vs Amazon's 13,752 nodes and Airports' 1,190 nodes). Per the project rubric,")
print("the two required additional analyses (Milestone 3) will be conducted on this")
print("dataset across all three GNN models.\n")

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='.')
split_idx = dataset.get_idx_split()

# 1. len
print(f"1. len(dataset) == {len(dataset)}")

# 2. Split sizes
train_idx = split_idx['train']
val_idx = split_idx['valid']
test_idx = split_idx['test']
print(f"2. Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

# 3. Class distribution
def get_dist(indices, name):
    subset = dataset[indices]
    labels = subset.y.view(-1)
    pos = int((labels == 1).sum().item())
    total = len(labels)
    neg = total - pos
    rate = pos / total * 100 if total > 0 else 0
    print(f"   {name}: label=0 count={neg}, label=1 count={pos}, positive rate={rate:.2f}%")
    return pos, neg

print("3. Class distribution per split:")
pos_train, neg_train = get_dist(train_idx, "Train")
pos_val, neg_val = get_dist(val_idx, "Val")
pos_test, neg_test = get_dist(test_idx, "Test")

# 4. mean +- std nodes/edges
num_nodes = [dataset[i].num_nodes for i in range(len(dataset))]
num_edges = [dataset[i].num_edges for i in range(len(dataset))]
print(f"4. Mean ± std of num_nodes per graph: {np.mean(num_nodes):.2f} ± {np.std(num_nodes):.2f}")
print(f"4. Mean ± std of num_edges per graph: {np.mean(num_edges):.2f} ± {np.std(num_edges):.2f}")

# 5. data[0].x.shape and data[0].edge_attr.shape
print(f"5. data[0].x.shape: {dataset[0].x.shape}, data[0].edge_attr.shape: {dataset[0].edge_attr.shape}")

# 6. pos_weight
pos_weight = neg_train / pos_train
print(f"6. pos_weight = num_negatives_train / num_positives_train = {pos_weight:.2f}")

# 7. Positive rate difference test
print("7. Does scaffold splitting cause positive rate to differ between train and test? Yes, scaffold splitting separates distinct chemical scaffolds into different sets, causing the positive class prevalence to vary.")

# 8. evaluator
evaluator = Evaluator(name='ogbg-molhiv')
print(f"8. evaluator.eval_metric = {evaluator.eval_metric}")

print("\nWHY BOTH ATOM FEATURES AND BOND STRUCTURE ARE NECESSARY:")
print("  Atom features (pharmacophores): identify specific functional groups such as")
print("  charged nitrogens required for chemical binding affinity. Without knowing")
print("  atom TYPE, the model cannot identify whether a binding site is present.")
print("  Bond structure (geometric fit): a linear chain vs a ring structure interacts")
print("  differently with the HIV enzyme pocket, even with identical atoms. Bond type")
print("  (single/double/aromatic) changes electron density and 3D geometry entirely.")
print("  A model using atoms only cannot reason about molecular shape; a model using")
print("  topology only cannot identify which atoms are pharmacologically active.")
print("  Both conditions must be simultaneously met for a molecule to inhibit HIV.")