import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import probplot

rng = np.random.default_rng(42)

n = 7000
B = 2000

# ------------------------------
# Funzioni
# ------------------------------
def bootstrap_means(x, B=1000, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    return x[idx].mean(axis=1)

def summarize_means(means, label):
    mean = np.mean(means)
    std = np.std(means, ddof=1)
    p2_5, p97_5 = np.percentile(means, [2.5, 97.5])
    skew = stats.skew(means)
    kurt = stats.kurtosis(means)
    return {
        "label": label,
        "mean": mean,
        "std": std,
        "p2.5": p2_5,
        "p97.5": p97_5,
        "skew": skew,
        "kurtosis": kurt
    }

def print_summary(summary):
    print(f"\n--- {summary['label']} ---")
    print(f"Mean: {summary['mean']:.4f}, Std: {summary['std']:.4f}")
    print(f"2.5%-97.5%: {summary['p2.5']:.4f} - {summary['p97.5']:.4f}")
    print(f"Skewness: {summary['skew']:.4f}, Kurtosis: {summary['kurtosis']:.4f}")

def plot_hist(means, label):
    plt.figure(figsize=(10,6))
    plt.hist(means, bins=60, density=True, alpha=0.7, edgecolor="black")
    mu, sigma = np.mean(means), np.std(means, ddof=1)
    x_vals = np.linspace(np.percentile(means,0.1), np.percentile(means,99.9), 300)
    plt.plot(x_vals, stats.norm.pdf(x_vals, mu, sigma), 'r--', lw=2, alpha=0.6, label="Gaussiana di riferimento")
    plt.axvline(np.percentile(means,2.5), color='orange', linestyle='--', lw=2, label='2.5% percentile')
    plt.axvline(np.percentile(means,97.5), color='orange', linestyle='--', lw=2, label='97.5% percentile')
    plt.title(f"Bootstrap means — {label}")
    plt.xlabel("bootstrap means")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------------------
# Bootstrap per diversi df e alpha
# ------------------------------
t_dfs = [0.5, 1.0, 2.0, 5.0, 20.0]
pareto_alphas = [0.4, 1.0, 2.0, 5.0, 20.0]

t_results = []
pareto_results = []

for df in t_dfs:
    x = rng.standard_t(df, size=n)
    m = bootstrap_means(x, B, rng)
    summary = summarize_means(m, f"t df={df}")
    t_results.append(summary)
    plot_hist(m, f"t df={df}")

for alpha in pareto_alphas:
    x = (1 - rng.random(n))**(-1/alpha)
    m = bootstrap_means(x, B, rng)
    summary = summarize_means(m, f"Pareto α={alpha}")
    pareto_results.append(summary)
    plot_hist(m, f"Pareto α={alpha}")

# ------------------------------
# Grafico Kurtosis vs Alpha (Pareto)
# ------------------------------
plt.figure(figsize=(8,5))
plt.plot(pareto_alphas, [r['kurtosis'] for r in pareto_results], 'o-', lw=2)
plt.xlabel("Alpha (Pareto)")
plt.ylabel("Excess Kurtosis")
plt.title("Kurtosis dei bootstrap means in funzione di α (Pareto)")
plt.grid(True)
plt.show()

# ------------------------------
# Grafico Skew vs df (t-Student)
# ------------------------------
plt.figure(figsize=(8,5))
plt.plot(t_dfs, [r['skew'] for r in t_results], 'o-', lw=2)
plt.xlabel("df (t-Student)")
plt.ylabel("Skewness")
plt.title("Skewness dei bootstrap means in funzione di df (t-Student)")
plt.grid(True)
plt.show()
