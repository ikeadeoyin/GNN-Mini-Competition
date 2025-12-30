# 🧪 The Hidden Molecule Challenge


**A Graph Neural Network Mini-Competition for GNNs for Rising Stars 2026**


![Status](https://img.shields.io/badge/Status-Active-success)
![Problem](https://img.shields.io/badge/Task-Graph_Classification-blue)
![Metric](https://img.shields.io/badge/Metric-Macro_F1-orange)

## 📌 Overview

Welcome to the **Hidden Molecule Challenge**! In early-stage drug discovery, identifying whether a molecule is mutagenic (causes DNA damage) is critical. Your goal is to build a Graph Neural Network (GNN) that can predict the mutagenicity of a molecule based solely on its atomic structure.

This challenge is designed to be solvable using methods covered in **DGL Lectures 1.1–4.6** (Message Passing, Pooling, etc.).

### The Task

* [cite_start]**Problem Type:** Graph Classification[cite: 217].

* **Input:** Small molecular graphs (Nodes = Atoms, Edges = Bonds).
* **Output:** Binary Label (0 = Safe, 1 = Mutagenic).
* **Difficulty:** The test set contains molecules with structural scaffolds not seen in the training set. Standard memorization will fail; your model must learn chemical rules.

---

## 📂 Repository Structure

[cite_start]The repository follows the standard competition format [cite: 234-248]:

```text
gnn-challenge/
├── data/
│   ├── train.csv               # Training IDs and Labels
│   ├── test.csv                # Test IDs (No labels - you predict these!)
│   ├── all_nodes.csv           # Features for all atoms in the dataset
│   └── all_edges.csv           # Bond connections for all molecules
├── starter_code/
│   ├── baseline.py             # Simple Random Forest baseline to get started
│   └── requirements.txt        # Python dependencies
├── submissions/                # Save your prediction CSVs here
├── scoring_script.py           # Script to evaluate your model locally
└── README.md


## 📊 Dataset Description
The dataset is derived from **MUTAG** (Mutagenic Compounds) but processed into a challenging scaffold split.

### 1. The Graph Data (`data/all_nodes.csv` & `data/all_edges.csv`)
Since graphs are complex, the raw structure is stored in two helper files:
* **Nodes:** `graph_id`, `node_id`, `atom_type` (The features).
* **Edges:** `graph_id`, `source_node`, `target_node`, `bond_type` (The adjacency).

### 2. The Split (`data/train.csv` & `data/test.csv`)
* **Train:** Contains `graph_id` and the ground truth `label` (0 or 1). Use this to train your GNN.
* **Test:** Contains only `graph_id`. The labels are hidden. **You must generate predictions for these IDs.**

## 🚀 Getting Started

### 1. Install Dependencies
Ensure you have Python installed, then install the required libraries:

```bash
pip install -r starter_code/requirements.txt
```

### 2. Run the Baseline Model

We provide a simple `baseline.py` that treats the graph as a "Bag of Atoms" (ignoring connections) and trains a Random Forest. This sets the score to beat.

```bash
python starter_code/baseline.py
```
Output: This will generate submissions/baseline_submission.csv.

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

## 💡 Tips for Success

Use GNNs: The baseline ignores bonds. Use GCNConv or GINConv (Lecture 3) to leverage the graph structure.

Global Pooling: Since this is Graph classification, remember to use a readout layer (e.g., global_mean_pool or global_add_pool) to aggregate node features into a graph embedding (Lecture 4).

Regularization: The dataset is small. Use Dropout and Weight Decay to prevent overfitting.

## 📜 License

MIT License.
