"""
Run this from your project root (same folder that contains the Data/ directory).
Paste the full output back to Claude.
"""
import pandas as pd
import numpy as np

for f in ["Data/shots_2023.csv", "Data/shots_2024.csv", "Data/shots_2025.csv"]:
    print(f"\n{'='*60}")
    print(f"FILE: {f}")
    try:
        df = pd.read_csv(f)
    except FileNotFoundError:
        print("  NOT FOUND — check filename"); continue

    print(f"  Total rows: {len(df):,}")
    print(f"  season values:        {df['season'].unique()}")
    print(f"  period dtype:         {df['period'].dtype}")
    print(f"  period unique values: {sorted(df['period'].unique())}")
    print(f"  homeTeamGoals dtype:  {df['homeTeamGoals'].dtype}")
    print(f"  awayTeamGoals dtype:  {df['awayTeamGoals'].dtype}")
    print(f"  teamCode dtype:       {df['teamCode'].dtype}")
    print(f"  homeTeamCode dtype:   {df['homeTeamCode'].dtype}")

    # Step 1
    s1 = df[df["period"] == 3]
    print(f"\n  [1] After period==3:           {len(s1):,} rows")

    # Step 2
    s2 = s1[abs(s1["homeTeamGoals"] - s1["awayTeamGoals"]) == 2].copy()
    print(f"  [2] After |goal diff|==2:      {len(s2):,} rows")
    if len(s2) > 0:
        diffs = (s2["homeTeamGoals"] - s2["awayTeamGoals"]).value_counts().to_dict()
        print(f"      Score diff breakdown: {diffs}")

    # Step 3
    if len(s2) > 0:
        s2["trailingTeam"] = np.where(
            s2["homeTeamGoals"] < s2["awayTeamGoals"],
            s2["homeTeamCode"], s2["awayTeamCode"]
        ).astype(str)
        s3 = s2[s2["teamCode"].astype(str) == s2["trailingTeam"]]
        print(f"  [3] After trailing team match: {len(s3):,} rows")
        if len(s2) > 0 and len(s3) == 0:
            print(f"      teamCode sample:     {s2['teamCode'].head(3).tolist()}")
            print(f"      trailingTeam sample: {s2['trailingTeam'].head(3).tolist()}")

    # Step 4
    if len(s2) > 0 and len(s3) > 0:
        s4 = s3[s3["time"] <= 1110]
        print(f"  [4] After pull cutoff:         {len(s4):,} rows")
        print(f"      time range: {s3['time'].min():.0f}–{s3['time'].max():.0f} seconds")