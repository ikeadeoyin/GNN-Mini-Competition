import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# 1. Load Data
# (Simple Baseline: Ignores graph structure, just counts atoms)
print("Loading data...")
nodes = pd.read_csv('data/train_nodes.csv')
labels = pd.read_csv('data/train_labels.csv')
test_nodes = pd.read_csv('data/test_nodes.csv')

def get_features(node_df):
    # Feature: Count how many of each atom type (0-6) exist in the molecule
    counts = node_df.groupby('graph_id')['atom_type'].value_counts().unstack(fill_value=0)
    return counts

X_train = get_features(nodes)
y_train = labels.set_index('graph_id').loc[X_train.index]['label']
X_test = get_features(test_nodes)

# Ensure columns match (fill missing atom types with 0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# 2. Train Model
print("Training Random Forest Baseline...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 3. Predict
preds = clf.predict(X_test)

# 4. Save Submission
submission = pd.DataFrame({'graph_id': X_test.index, 'label': preds})
submission.to_csv('submissions/baseline_submission.csv', index=False)
print("✅ Baseline submission saved!")