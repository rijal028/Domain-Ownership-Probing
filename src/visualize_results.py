import os
import numpy as np
import matplotlib.pyplot as plt

# ======================================
# SAVE HELPERS
# ======================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_fig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

# ======================================
# PLOT 1: HEATMAP MATRIX
# ======================================
def plot_similarity_heatmap(domain_names, sim_matrix, save_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(sim_matrix, interpolation="nearest")
    plt.xticks(range(len(domain_names)), domain_names, rotation=45, ha="right")
    plt.yticks(range(len(domain_names)), domain_names)

    plt.colorbar(label="Cosine Similarity")
    plt.title("Domain Similarity Matrix (Prototype Cosine Similarity)")
    save_fig(save_path)

# ======================================
# PLOT 2: WIN-RATE BAR
# ======================================
def plot_winrate_bar(domain_names, win_rates, save_path):
    plt.figure(figsize=(10, 5))
    plt.bar(domain_names, win_rates)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Win Rate")
    plt.ylim(0, 1.0)
    plt.title("Domain Ownership (Win Rate)")
    save_fig(save_path)

# ======================================
# PLOT 3: COHESION BAR
# ======================================
def plot_cohesion_bar(domain_names, cohesion_avgs, save_path):
    plt.figure(figsize=(10, 5))
    plt.bar(domain_names, cohesion_avgs)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Avg Similarity to Prototype")
    plt.ylim(0, 1.0)
    plt.title("Domain Cohesion (Intra-domain Compactness)")
    save_fig(save_path)

# ======================================
# MAIN ENTRY
# ======================================
def export_all_figures(results, out_dir="outputs/figures"):
    """
    results format (from your main probe):
    {
      "domain_names": [...],
      "similarity_matrix": np.array NxN,
      "win_rates": {domain: float},
      "cohesion_avg": {domain: float},
    }
    """
    ensure_dir(out_dir)

    domain_names = results["domain_names"]
    sim_matrix = results["similarity_matrix"]

    win_rates = [results["win_rates"][d] for d in domain_names]
    cohesion_avgs = [results["cohesion_avg"][d] for d in domain_names]

    plot_similarity_heatmap(
        domain_names,
        sim_matrix,
        os.path.join(out_dir, "heatmap.png")
    )

    plot_winrate_bar(
        domain_names,
        win_rates,
        os.path.join(out_dir, "ownership_bar.png")
    )

    plot_cohesion_bar(
        domain_names,
        cohesion_avgs,
        os.path.join(out_dir, "cohesion_bar.png")
    )

    print(f"[OK] Figures saved to: {out_dir}")
    print("- heatmap.png")
    print("- ownership_bar.png")
    print("- cohesion_bar.png")