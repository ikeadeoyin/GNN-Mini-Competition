import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
from sklearn.model_selection import train_test_split

# Load data
nodes_df = pd.read_csv("data/train_nodes.csv")
edges_df = pd.read_csv("data/train_edges.csv")
labels_df = pd.read_csv("data/train_labels.csv")

# Get feature dimensions for one-hot encoding
num_atom_types = nodes_df.atom_type.max() + 1
num_bond_types = edges_df.bond_type.max() + 1
print(f"Atom types: {num_atom_types}, Bond types: {num_bond_types}")


def build_graph_from_id(gid, nodes_df, edges_df, labels_df, num_atom_types, num_bond_types):
    """Build a PyG Data object from a graph ID."""
    nodes_g = nodes_df[nodes_df.graph_id == gid]
    node_ids = nodes_g.node_id.values
    id_map = {nid: i for i, nid in enumerate(node_ids)}

    # One-hot encode atom types
    atom_types = nodes_g.atom_type.values
    x = torch.zeros(len(atom_types), num_atom_types)
    x[torch.arange(len(atom_types)), atom_types] = 1.0

    # Build edge index
    edges_g = edges_df[edges_df.graph_id == gid]
    src = edges_g.source_node.map(id_map).values
    dst = edges_g.target_node.map(id_map).values
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
    
    # One-hot encode bond types as edge features
    bond_types = edges_g.bond_type.values
    edge_attr = torch.zeros(len(bond_types), num_bond_types)
    edge_attr[torch.arange(len(bond_types)), bond_types] = 1.0

    label = labels_df[labels_df.graph_id == gid].label.values[0]

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(label, dtype=torch.long)
    )


def build_test_graph(gid, nodes_df, edges_df, num_atom_types, num_bond_types):
    """Build a PyG Data object for test graphs (no labels)."""
    nodes_g = nodes_df[nodes_df.graph_id == gid]
    node_ids = nodes_g.node_id.values
    id_map = {nid: i for i, nid in enumerate(node_ids)}
    
    # One-hot encode atom types
    atom_types = nodes_g.atom_type.values
    x = torch.zeros(len(atom_types), num_atom_types)
    x[torch.arange(len(atom_types)), atom_types] = 1.0
    
    # Build edge index
    edges_g = edges_df[edges_df.graph_id == gid]
    src = edges_g.source_node.map(id_map).values
    dst = edges_g.target_node.map(id_map).values
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
    
    # One-hot encode bond types
    bond_types = edges_g.bond_type.values
    edge_attr = torch.zeros(len(bond_types), num_bond_types)
    edge_attr[torch.arange(len(bond_types)), bond_types] = 1.0
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class MoleculeGINE(nn.Module):
    """Graph Isomorphism Network with Edge features for molecular classification."""
    
    def __init__(self, in_feats, edge_feats, hidden_feats, num_classes):
        super().__init__()
        
        self.edge_emb = nn.Linear(edge_feats, hidden_feats)
        
        self.conv1 = GINEConv(
            nn.Sequential(nn.Linear(in_feats, hidden_feats), nn.ReLU(), nn.Linear(hidden_feats, hidden_feats)),
            edge_dim=hidden_feats
        )
        self.bn1 = nn.BatchNorm1d(hidden_feats)
        
        self.conv2 = GINEConv(
            nn.Sequential(nn.Linear(hidden_feats, hidden_feats), nn.ReLU(), nn.Linear(hidden_feats, hidden_feats)),
            edge_dim=hidden_feats
        )
        self.bn2 = nn.BatchNorm1d(hidden_feats)
        
        self.conv3 = GINEConv(
            nn.Sequential(nn.Linear(hidden_feats, hidden_feats), nn.ReLU(), nn.Linear(hidden_feats, hidden_feats)),
            edge_dim=hidden_feats
        )
        self.bn3 = nn.BatchNorm1d(hidden_feats)
        
        self.dropout = nn.Dropout(0.5)
        self.classify = nn.Linear(hidden_feats, num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        edge_emb = self.edge_emb(edge_attr)
        
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_emb)))
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_emb)))
        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_emb)))

        hg = global_add_pool(x, batch)
        return self.classify(self.dropout(hg))


# Build training graphs
graphs = [build_graph_from_id(gid, nodes_df, edges_df, labels_df, num_atom_types, num_bond_types) 
          for gid in labels_df.graph_id.values]
print(f"Built {len(graphs)} training graphs")

# Train/val split
train_graphs, val_graphs = train_test_split(
    graphs, test_size=0.2, random_state=42, stratify=[g.y.item() for g in graphs]
)
train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=8, shuffle=False)
print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")

# Initialize model
model = MoleculeGINE(in_feats=num_atom_types, edge_feats=num_bond_types, hidden_feats=64, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training loop with best model tracking
best_val_acc = 0
best_model_state = None

for epoch in range(50):
    model.train()
    total_loss = 0
    for batch in train_loader:
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = loss_fn(logits, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            preds = logits.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total += batch.y.size(0)
    
    val_acc = correct / total
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Val Acc: {val_acc:.4f} *BEST*")
    else:
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Val Acc: {val_acc:.4f}")

# Load best model
print(f"\nLoading best model (val acc: {best_val_acc:.4f})")
model.load_state_dict(best_model_state)

# Generate test predictions
test_nodes_df = pd.read_csv("data/test_nodes.csv")
test_edges_df = pd.read_csv("data/test_edges.csv")
test_graph_ids = test_nodes_df.graph_id.unique()
print(f"Test graphs: {len(test_graph_ids)}")

test_graphs = [(gid, build_test_graph(gid, test_nodes_df, test_edges_df, num_atom_types, num_bond_types)) 
               for gid in test_graph_ids]

model.eval()
predictions = []
with torch.no_grad():
    for gid, g in test_graphs:
        batch = torch.zeros(g.num_nodes, dtype=torch.long)
        logits = model(g.x, g.edge_index, g.edge_attr, batch)
        pred = logits.argmax(dim=1).item()
        predictions.append((gid, pred))

# Save submission
submission_df = pd.DataFrame(predictions, columns=['graph_id', 'label']).sort_values('graph_id')
submission_df.to_csv("submissions/initial_submission.csv", index=False)
print(f"\nSubmission saved to submissions/initial_submission.csv")
print(submission_df.head(10))
