import torch, numpy as np, random, time, json, os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import (cohen_kappa_score, f1_score,
                             precision_score, recall_score)
torch.manual_seed(42); np.random.seed(42); random.seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    # Print environment versions
    import sys
    import torch_geometric
    print("Python version:", sys.version.replace("\n", " "))
    print("PyTorch version:", torch.__version__)
    print("torch_geometric version:", torch_geometric.__version__)
    print("CUDA available:", torch.cuda.is_available())

    from torch_geometric.datasets import Airports
    from scipy.stats import spearmanr
    from collections import deque

    dataset = Airports(root=".", name="USA")
    data = dataset[0]

    num_nodes = data.num_nodes
    edge_index = data.edge_index
    num_edges = data.num_edges
    x = data.x
    labels = data.y.view(-1).to(torch.long)

    # 1) Confirm specs
    print(f"\n1. Confirm num_nodes={num_nodes}, num_edges={num_edges}, x.shape=[{x.shape[0]},{x.shape[1]}]")

    # 2) Label distribution: count + % per class 0-3
    print("\n2. Label distribution: count + % per class 0–3")
    total = int(labels.numel())
    for c in range(4):
        cnt = int((labels == c).sum().item())
        pct = 100.0 * cnt / total
        print(f"class {c}: count={cnt} ({pct:.2f}%)")

    # 3) Degree stats: mean, median, max, min, std
    print("\n3. Degree stats: mean, median, max, min, std")
    row, col = edge_index[0], edge_index[1]

    # Determine whether edge_index is stored symmetrically.
    # If symmetric: degrees = bincount(row) (neighbors counted once per outgoing direction).
    # If not symmetric: treat as undirected and count incident edges for both endpoints.
    pairs = list(zip(row.tolist(), col.tolist()))
    pair_set = set(pairs)
    sample_n = min(200, len(pairs))
    sample_idxs = list(range(sample_n))
    symmetric_hits = 0
    for i in sample_idxs:
        u, v = pairs[i]
        if (v, u) in pair_set:
            symmetric_hits += 1
    symmetric_ratio = symmetric_hits / max(1, sample_n)
    is_symmetric = symmetric_ratio >= 0.9

    if is_symmetric:
        degrees = torch.bincount(row, minlength=num_nodes)
    else:
        degrees = torch.bincount(torch.cat([row, col], dim=0), minlength=num_nodes)

    degrees_np = degrees.to(torch.float).cpu().numpy()
    mean_deg = float(degrees_np.mean())
    median_deg = float(np.median(degrees_np))
    max_deg = int(degrees_np.max())
    min_deg = int(degrees_np.min())
    std_deg = float(degrees_np.std(ddof=0))
    print(f"mean={mean_deg:.4f}, median={median_deg:.4f}, max={max_deg}, min={min_deg}, std={std_deg:.4f}")

    # 4) Top-10 highest-degree airports: index + degree
    print("\n4. Top-10 highest-degree airports: index + degree")
    top_vals, top_idx = degrees.topk(10, largest=True, sorted=True)
    for nidx, deg in zip(top_idx.tolist(), top_vals.tolist()):
        print(f"{nidx}: {deg}")

    # 5) Spearman correlation (degree vs label)
    print("\n5. Spearman correlation (degree vs label)")
    # Spearman expects 1D arrays of equal length
    corr, pval = spearmanr(degrees_np, labels.cpu().numpy())
    print(f"corr={corr}")
    print(f"pval={pval}")
    if pval < 0.05:
        sig_sentence = "This indicates a statistically significant monotonic relationship at alpha=0.05."
    else:
        sig_sentence = "This does not indicate statistical significance at alpha=0.05, though a monotonic trend may still exist."
    print(sig_sentence)

    # 6) Graph connectivity: single connected component?
    print("\n6. Graph connectivity: single connected component?")
    # Build undirected adjacency for BFS
    adj = [[] for _ in range(num_nodes)]
    for u, v in pairs:
        adj[u].append(v)
        adj[v].append(u)

    visited = [False] * num_nodes
    comp_sizes = []
    for start in range(num_nodes):
        if not visited[start]:
            q = deque([start])
            visited[start] = True
            size = 0
            while q:
                u = q.popleft()
                size += 1
                for w in adj[u]:
                    if not visited[w]:
                        visited[w] = True
                        q.append(w)
            comp_sizes.append(size)

    num_components = len(comp_sizes)
    is_connected = (num_components == 1)
    print(f"single_connected_component? {is_connected} (num_components={num_components})")

    # Print M1 justification + formulation rationale block (exact wording/structure)
    print(
        "WHY STRUCTURE IS THE ONLY SIGNAL:\n"
        "  One-hot identity features are arbitrary node IDs — an MLP trained on them\n"
        "  cannot generalise to predict activity for any node. Only graph-structural\n"
        "  position (neighbourhood connectivity via message passing) is transferable.\n"
        "\n"
        "ORDINAL vs CLASSIFICATION FORMULATION:\n"
        "  Original paper treats labels 0,1,2,3 as independent unordered classes.\n"
        "  CrossEntropyLoss penalises predicting 0 for a true 3 hub IDENTICALLY to\n"
        "  predicting 2 for a true 3 hub — the natural ordering 0<1<2<3 is ignored.\n"
        "  Our ordinal approach uses MSELoss on a scalar: predicting 0 for a true 3\n"
        "  incurs (3−0)²=9 vs (2−3)²=1, penalising large ordinal violations quadratically.\n"
        f"  Spearman corr = {corr} (p={pval}): labels have continuous monotonic structure\n"
        "  that justifies treating them as an ordered scale rather than categories.\n"
        "  Both formulations will be trained in P2 to empirically validate this claim."
    )

    # Splits (70/15/15, seed=42) and save to splits_usa.pt
    print("\nSplits (70/15/15, seed=42)")
    generator = torch.Generator().manual_seed(42)
    idx = torch.randperm(1190, generator=generator)
    train_idx = idx[:833]
    val_idx = idx[833:1012]
    test_idx = idx[1012:]
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    out_path = "splits_usa.pt"
    torch.save(splits, out_path)
    print(f"Saved splits to: {out_path}")

    def print_split_dist(name, split_idx):
        y = labels[split_idx]
        total_s = int(y.numel())
        print(f"\n{name}: num_samples={total_s}")
        for c in range(4):
            cnt = int((y == c).sum().item())
            pct = 100.0 * cnt / total_s
            print(f"  class {c}: count={cnt} ({pct:.2f}%)")

    print_split_dist("train", train_idx)
    print_split_dist("val", val_idx)
    print_split_dist("test", test_idx)


if __name__ == "__main__":
    main()

