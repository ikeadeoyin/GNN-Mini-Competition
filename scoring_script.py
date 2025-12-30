import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import sys

# Usage: python scoring_script.py submissions/sample_submission.csv

def score_submission(submission_file):
    # 1. Check if file exists
    try:
        submission = pd.read_csv(submission_file)
    except FileNotFoundError:
        print(f"Error: File '{submission_file}' not found.")
        return

    # 2. Load the Hidden Ground Truth
    # This file contains the real labels for the test set
    try:
        ground_truth = pd.read_csv('ground_truth.csv')
    except FileNotFoundError:
        print("Error: 'ground_truth.csv' not found. Make sure you generated the data.")
        return

    # 3. Align the Data
    # We merge on 'graph_id' to ensure we are comparing the correct predictions
    # inner join ensures we only score IDs that exist in both files
    merged = pd.merge(ground_truth, submission, on='graph_id', suffixes=('_true', '_pred'))

    # 4. Check for Missing Predictions
    if len(merged) != len(ground_truth):
        print(f"Warning: The submission contains {len(merged)} predictions, but the test set has {len(ground_truth)}.")

    # 5. Compute Metrics
    # Accuracy is standard for MUTAG, but F1 is requested by the prompt for 'difficulty'
    acc = accuracy_score(merged['label_true'], merged['label_pred'])
    f1 = f1_score(merged['label_true'], merged['label_pred'], average='macro')

    # 6. Print Results
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")

if __name__ == "__main__":
    # Ensure a submission file was provided in the command line
    if len(sys.argv) < 2:
        print("Usage: python scoring_script.py <path_to_submission_csv>")
    else:
        score_submission(sys.argv[1])