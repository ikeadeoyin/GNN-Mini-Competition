import sys
import pandas as pd
from pathlib import Path

# 1. Parse Arguments
if len(sys.argv) < 3:
    print("Usage: python update_leaderboard.py <username> <score>")
    sys.exit(1)

username = sys.argv[1]
try:
    score = float(sys.argv[2])
except ValueError:
    print(f"Error: Score '{sys.argv[2]}' is not a valid number.")
    sys.exit(1)

lb_path = Path("leaderboard.md")
rows = []

# 2. Read existing leaderboard
if lb_path.exists():
    with open(lb_path, "r") as f:
        for line in f:
            stripped = line.strip()
            # Skip empty lines, headers ("Rank"), or separator lines (lines with only dashes/pipes)
            if not stripped.startswith("|") or "Rank" in line:
                continue
            
            # Check if it's a separator line (e.g. |---|---|)
            # We remove pipes and spaces; if only dashes remain, it's a separator.
            if set(stripped.replace("|", "").replace(" ", "")) == {"-"}:
                continue

            try:
                # Extract columns: | Rank | Username | Score |
                parts = [p.strip() for p in line.split("|")]
                # parts[0] is empty (before first pipe), parts[1] is Rank, parts[2] is Username, parts[3] is Score
                if len(parts) < 4: 
                    continue
                    
                row_user = parts[2]
                row_score = float(parts[3])
                
                rows.append({
                    "Username": row_user,
                    "Score": row_score,
                })
            except ValueError:
                continue

# 3. Add new score
rows.append({"Username": username, "Score": score})

# 4. Sort and Clean (Keep highest score per user)
df = pd.DataFrame(rows)
if not df.empty:
    df = (
        df.sort_values("Score", ascending=False)
          .drop_duplicates("Username", keep="first")
          .reset_index(drop=True)
    )
    df["Rank"] = df.index + 1
else:
    # Handle edge case if file was empty
    df = pd.DataFrame([{"Rank": 1, "Username": username, "Score": score}])

# 5. Write back to file
with open(lb_path, "w") as f:
    f.write("# 🏆 GNN Mini-Challenge Leaderboard\n\n")
    f.write("| Rank | Username | Score |\n")
    f.write("|:----:|:--------:|:-----:|\n")  # Standard Markdown alignment
    for _, r in df.iterrows():
        f.write(f"| {r['Rank']} | {r['Username']} | {r['Score']:.4f} |\n")
    f.write("\n_Last updated automatically._\n")

print(f"✅ Leaderboard updated for {username} with score {score}")