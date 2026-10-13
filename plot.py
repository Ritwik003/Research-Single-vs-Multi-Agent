"""
Plot the headline chart: ACCURACY vs TOKENS (compute).
Run research.py first (it produces results.csv), then run this.

The chart is the whole argument in one picture:
  - x-axis = tokens spent (compute cost)
  - y-axis = accuracy
  - one point per system
A system that sits UP and to the LEFT is better (more accuracy, less compute).
Comparing 'multi' and 'single_vote' at similar x answers the structure question.
"""

import csv
import matplotlib.pyplot as plt

# read results
rows = []
with open("results.csv") as f:
    for r in csv.DictReader(f):
        r["correct"] = int(r["correct"])
        r["tokens"] = float(r["tokens"])
        rows.append(r)

systems = ["single", "single_vote", "multi"]
labels = {"single": "Single Agent",
          "single_vote": "Single + Self-Consistency",
          "multi": "Multi-Agent (Gen+Critic)"}
colors = {"single": "#1f4e79", "single_vote": "#e07a5f", "multi": "#2a9d8f"}

fig, ax = plt.subplots(figsize=(8, 6))

for s in systems:
    srows = [r for r in rows if r["system"] == s]
    n = len(srows)
    acc = sum(r["correct"] for r in srows) / n * 100
    tok = sum(r["tokens"] for r in srows) / n
    ax.scatter(tok, acc, s=260, color=colors[s], zorder=3,
               edgecolors="white", linewidths=2)
    ax.annotate(f"{labels[s]}\n{acc:.0f}% @ {tok:.0f} tok",
                (tok, acc), textcoords="offset points", xytext=(12, 8),
                fontsize=10, fontweight="bold")

ax.set_xlabel("Compute cost  (average tokens per problem)", fontsize=12)
ax.set_ylabel("Accuracy  (%)", fontsize=12)
ax.set_title("Multi-Agent vs Single-Agent at Equal Compute",
             fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig("result_chart.png", dpi=150)
print("Saved result_chart.png")
