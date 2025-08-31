import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

rng = np.random.default_rng(42)

n = 10**4        # sample size
B = 4000        # bootstrap replicates
bins = 60       # histogram bins

# ------------------------------
# 1) Funzioni di servizio
# ------------------------------
def bootstrap_means(x, B=1000, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    return x[idx].mean(axis=1)

def summarize_bootstrap(means, label, original=None):
    boot_mean = float(np.mean(means))
    boot_std = float(np.std(means, ddof=1))
    summary = {
        "label": label,
        "boot_mean_of_means": boot_mean,
        "boot_se": boot_std,
        "p2.5": float(np.percentile(means, 2.5)),
        "p97.5": float(np.percentile(means, 97.5)),
        "skew": float(stats.skew(means)),
        "kurtosis_excess": float(stats.kurtosis(means)),
    }
    if original is not None:
        l = len(original)
        sd = np.std(original, ddof=1)
        summary.update({
            "original_mean": float(np.mean(original)),
            "original_std": float(sd),
            "std_theoretical": float(sd / np.sqrt(l))
        })
    return summary

# Stampa ordinata su console
def print_summary(summary):
    def fmt(v):
        return f"{v:.4f}" if v is not None else "—"

    # Lista chiave + etichetta in ordine
    keys = [
        ("original_mean",   "Mean (original)"),
        ("original_std",    "Std (original)"),
        ("boot_mean_of_means", "Mean (boot)"),
        ("boot_se",         "SE (boot)"),
        ("p2.5",            "2.5%"),
        ("p97.5",           "97.5%"),
        ("skew",            "Skew"),
        ("kurtosis_excess", "Kurtosis (ex)"),
        ("std_theoretical", "Std theoretical")
    ]

    print(f"\n--- {summary['label']} ---")
    for key, label in keys:
        print(f"{label:<20} {fmt(summary.get(key))}")

# Plot come prima
def plot_means(title, means, bins=60):
    plt.figure(figsize=(12, 8))
    plt.hist(means, bins=bins, density=True, alpha=0.7, edgecolor="black")

    mu, sigma = np.mean(means), np.std(means, ddof=1)
    x_vals = np.linspace(np.percentile(means,0.1), np.percentile(means,99.9), 300)
    plt.plot(x_vals, norm.pdf(x_vals, mu, sigma), 'r--', lw=2, alpha=0.6, label="Gaussiana di riferimento")

    # percentili
    plt.axvline(np.percentile(means, 2.5), color='orange', linestyle='--', lw=2, label='2.5% percentile')
    plt.axvline(np.percentile(means, 97.5), color='orange', linestyle='--', lw=2, label='97.5% percentile')

    plt.title(f"Bootstrap means — {title}")
    plt.xlabel("bootstrap means")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------------------
# 2) Lista esperimenti
# ------------------------------
experiments = []

# Normal
x = rng.normal(0,1,n)
m = bootstrap_means(x,B,rng)
experiments.append((m, summarize_bootstrap(m,"Normal(0,1)",original=x)))

# t-Student
for nu in [0.4, 1.0, 5.0, 20.0]:
    x = rng.standard_t(df=nu, size=n)
    m = bootstrap_means(x,B,rng)
    desc = f"t-Student(df={nu})"
    if nu < 1: desc += " no mean"
    elif nu < 2: desc += " inf var"
    elif nu < 15: desc += " finite var"
    else: desc += " ~Normal"
    experiments.append((m, summarize_bootstrap(m,desc,original=x)))

# Pareto
def pareto_sample(alpha,size,rng):
    return (1 - rng.random(size))**(-1/alpha)
for alpha in [0.4,1.0,5.0,20.0]:
    x = pareto_sample(alpha,n,rng)
    m = bootstrap_means(x,B,rng)
    desc = f"Pareto(alpha={alpha})"
    if alpha < 1: desc += " no mean"
    elif alpha < 2: desc += " inf var"
    elif alpha < 15: desc += " finite var"
    else: desc += " ~Normal"
    experiments.append((m, summarize_bootstrap(m,desc,original=x)))

# ------------------------------
# 3) Esegui esperimenti
# ------------------------------
for means, summary in experiments:
    print_summary(summary)             # stampa ordinata su console
    plot_means(summary["label"], means)  # plot uguale a prima
