#!/usr/bin/env python
import pandas as pd, matplotlib.pyplot as plt, os
df = pd.read_csv("results/metrics.csv")
os.makedirs("results/charts", exist_ok=True)
plt.figure(); df.groupby("method")["ppl"].mean().plot(kind="bar", title="PPL (lower is better)"); plt.tight_layout(); plt.savefig("results/charts/accuracy_vs_size.png"); plt.close()
plt.figure(); df.groupby("method")["throughput_tok_s"].mean().plot(kind="bar", title="Throughput (tok/s)"); plt.tight_layout(); plt.savefig("results/charts/speed_vs_ram.png"); plt.close()
print("Saved charts to results/charts/")
