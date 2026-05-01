# GNN Empirical Comparison: Node, Graph, and Ordinal Tasks

This repository contains the full experimental pipeline, analysis scripts, and IEEE-formatted research manuscript for a systematic comparison of three Graph Neural Network (GNN) architectures: **GAT**, **GraphSAGE**, and **GIN**.

## 🚀 Overview

The project evaluates architectural expressivity and efficiency across three fundamentally different graph-learning paradigms:
1.  **Amazon Computers**: 10-class node classification on a high-density co-purchase graph.
2.  **USA Airports**: Ordinal node regression on a traffic network, comparing `MSELoss` vs `CrossEntropyLoss`.
3.  **OGBG-MOLHIV**: Binary graph classification on 41,127 molecular graphs with Virtual Node enhancements.

## 📊 Key Research Contributions

-   **Ordinal Regression Breakthrough**: Demonstrated that using an ordinal formulation (`MSELoss`) on the Airports dataset eliminates 100% of catastrophic regional-to-hub misclassifications, reducing the catastrophic error rate by **2.381 percentage points**.
-   **Architecture Ranking**: GIN emerged as the most consistently expressive architecture overall (Rank Sum = 5), followed by GAT (6) and GraphSAGE (7).
-   **Efficiency-Accuracy Trade-off**: Identified GraphSAGE as the most efficient model for large-scale screening, delivering competitive results with **<50% of the parameter count** of GAT/GIN.
-   **Ablation Insights**: Confirmed that molecular inhibition prediction requires both atom features and bond topology; removing bond information causes a performance drop of up to **0.169 ROC-AUC**.

## 🛠️ Repository Structure

```text
├── amazon/               # Logs and results for Amazon Computers task
├── airports/             # Logs and results for USA Airports ordinal task
├── molhiv/               # Logs and results for OGBG-MOLHIV graph task
├── figures/              # Intermediate training and t-SNE visualizations
├── p10_part1.py          # Data extraction and per-epoch normalization script
├── p10_part2.py          # Cross-dataset visualization and plotting script
├── paper.tex             # Main IEEE conference manuscript (LaTeX)
└── p10_data.json         # Intermediate structured data for analysis
```

## ⚙️ Usage & Analysis Pipeline

To regenerate the cross-dataset analysis, figures, and consolidated output:

1.  **Extract & Normalize Results**:
    ```bash
    python p10_part1.py
    ```
    *This script parses the logs from all three datasets, normalizes all training times to "per-epoch" values for fair comparison, and generates `p10_data.json`.*

2.  **Generate Visualizations**:
    ```bash
    python p10_part2.py
    ```
    *This script generates the high-resolution figures used in the paper, including the rank heatmap and the log-scaled training time comparison.*

## 📈 Experimental Results Summary

| Dataset | Best Architecture | Primary Metric | Per-Epoch Time |
| :--- | :--- | :--- | :--- |
| **Amazon** | GAT | 90.99% Acc | 0.075s |
| **Airports** | GIN | 0.5758 Kappa | 0.005s |
| **MOLHIV** | GIN | 0.7842 ROC | 11.821s |

## 📝 Authors
- **Adithya Nangarath**
- **K V Nikhilesh**
- **Vishruth Vijay**
