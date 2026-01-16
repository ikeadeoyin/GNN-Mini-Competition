# 🧪 The Hidden Molecule Challenge


**A Graph Neural Network Mini-Competition for GNNs for Rising Stars 2026**


![Status](https://img.shields.io/badge/Status-Active-success)
![Problem](https://img.shields.io/badge/Task-Graph_Classification-blue)
![Metric](https://img.shields.io/badge/Metric-Macro_F1-orange)

## 📌 Overview

Welcome to the **Hidden Molecule Challenge**! In early-stage drug discovery, identifying whether a molecule is mutagenic (causes DNA damage) is critical. Your goal is to build a Graph Neural Network (GNN) that can predict the mutagenicity of a molecule based solely on its atomic structure.

This challenge is designed to be solvable using methods covered in **DGL Lectures 1.1–4.6** (Message Passing, Pooling, etc.).

### The Task

* **Problem Type:** Graph Classification

* **Input:** Small molecular graphs (Nodes = Atoms, Edges = Bonds).
* **Output:** Binary Label (0 = Safe, 1 = Mutagenic).
* **Difficulty:** The test set contains molecules with structural scaffolds not seen in the training set. Standard memorization will fail; your model must learn chemical rules.

---

## 📂 Repository Structure

[cite_start]The repository follows the standard competition format [cite: 234-248]:

```text
gnn-challenge/
├── data/
│   ├── train_nodes.csv         # Node features for training graphs
│   ├── train_edges.csv         # Edge connections for training graphs
│   ├── train_labels.csv        # Training graph IDs and labels
│   ├── test_nodes.csv          # Node features for test graphs
│   └── test_edges.csv          # Edge connections for test graphs
├── starter_code/
│   ├── baseline.py                # GNN baseline using GINEConv with edge features
│   └── requirements.txt        # Python dependencies
├── submissions/                # Save your prediction CSVs here
├── scoring_script.py           # Script to evaluate your model locally
├── leaderboard.md              # Current standings
└── README.md
```

## 📊 Dataset Description
The dataset is derived from **MUTAG** (Mutagenic Compounds) but processed into a challenging scaffold split.

### 1. Training Data
* **`train_nodes.csv`:** `graph_id`, `node_id`, `atom_type` - Node features for each atom
* **`train_edges.csv`:** `graph_id`, `source_node`, `target_node`, `bond_type` - Bond connections
* **`train_labels.csv`:** `graph_id`, `label` - Ground truth labels (0 = Safe, 1 = Mutagenic)

### 2. Test Data
* **`test_nodes.csv`:** Node features for test molecules
* **`test_edges.csv`:** Bond connections for test molecules
* **Note:** Test labels are hidden. You must predict these!

### 3. Feature Details
* **`atom_type`:** 7 unique atom types (0-6), representing different elements
* **`bond_type`:** 4 unique bond types (0-3), representing single, double, aromatic bonds, etc.

## 🚀 Getting Started

### 1. Install Dependencies
Ensure you have Python installed, then install the required libraries:

```bash
pip install -r starter_code/requirements.txt
```

### 2. Run the Baseline Model

We provide a GNN baseline model using GINEConv:

```bash
python starter_code/baseline.py
```

This model leverages both node features (atom types) and edge features (bond types) using Graph Isomorphism Network with Edge features.

**Baseline Performance:**
| Metric | Score |
|--------|-------|
| Val Accuracy | ~90% |
| Test Accuracy | ~79% |

🎯 **Your goal:** Improve upon this baseline by experimenting with architectures, hyperparameters, and regularization!

### 3. Evaluate Your Model

You can check your performance locally using the scoring script. (Note: In a real hosted version, the ground truth would be hidden, but for this repo, it checks against the local truth file).

```bash
python scoring_script.py submissions/baseline_submission.csv
```

Target Metric: Macro F1 Score.

## 📝 Submission Format

Your submission file must be a CSV with exactly two columns and a header row:

```csv
graph_id,label
15,0
23,1
42,0
...
```

graph_id: Must match the IDs in data/test.csv.

label: Your prediction (0 or 1).

## 📤 How to Submit

- Fork this repository.
- Save your prediction as `submissions/YourName.csv`.
- Open a Pull Request to the main branch.


## 🏆 Leaderboard

<!-- todo -->

---

## 💡 Tips for Success

1. **Use GNNs:** The baseline ignores bonds. Use `GCNConv`, `GINConv`, or `GINEConv` to leverage the graph structure.

2. **Don't forget edge features!** The `bond_type` column contains valuable information about chemical bonds. Models like `GINEConv` can incorporate edge features.

3. **Global Pooling:** Since this is graph classification, use a readout layer (`global_mean_pool` or `global_add_pool`) to aggregate node features into a graph embedding.

4. **One-hot encoding:** Convert categorical features (`atom_type`, `bond_type`) to one-hot vectors for better performance.

5. **Regularization:** The dataset is small (~150 training graphs). Use Dropout and Weight Decay to prevent overfitting.

6. **Early stopping:** Track validation accuracy and save the best model checkpoint.

---

## 📜 License

MIT License.
