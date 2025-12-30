import sys
import re
import pandas as pd
from pathlib import Path

username = sys.argv[1]
score = float(sys.argv[2])

lb_path = Path("leaderboard.md")

rows = []

if lb_path.exists():
    with open(lb_path) as f:
        for line in f:
            if line.startswith("|") and "Rank" not in line and "—" not in line:
                parts = [p.strip() for p in line.split("|")[1:-1]]
                rows.append({
                    "Username": parts[1],
                    "Score": float(parts[2]),
                })

rows.append({"Username": username, "Score": score})

df = pd.DataFrame(rows)
df = (
    df.sort_values("Score", ascending=False)
      .drop_duplicates("Username", keep="first")
      .reset_index(drop=True)
)
df["Rank"] = df.index + 1

with open(lb_path, "w") as f:
    f.write("# 🏆 GNN Mini-Challenge Leaderboard\n\n")
    f.write("| Rank | Username | Score |\n")
    f.write("|------|----------|-------|\n")
    for _, r in df.iterrows():
        f.write(f"| {r['Rank']} | {r['Username']} | {r['Score']:.4f} |\n")
    f.write("\n_Last updated automatically._\n")
