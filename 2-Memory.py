import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import toeplitz, cholesky

rng = np.random.default_rng(42)

n = 7000        # sample size
B = 2000        # bootstrap replicates
bins = 60       # histogram bins

# ------------------------------  
# Funzioni di servizio  
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

# ------------------------------  
# Stampa ordinata su console  
# ------------------------------  
def print_summary(summary):
    def fmt(v):
        return f"{v:.4f}" if v is not None else "—"

    keys = [
        ("original_mean", "Mean (original)"),
        ("original_std", "Std (original)"),
        ("boot_mean_of_means", "Mean (boot)"),
        ("boot_se", "SE (boot)"),
        ("p2.5", "2.5%"),
        ("p97.5", "97.5%"),
        ("skew", "Skew"),
        ("kurtosis_excess", "Kurtosis (ex)"),
        ("std_theoretical", "Std theoretical")
    ]

    print(f"\n--- {summary['label']} ---")
    for key, label in keys:
        print(f"{label:<20} {fmt(summary.get(key))}")

# ------------------------------  
# Plot dei bootstrap  
# ------------------------------  
def plot_means(title, means, bins=60):
    plt.figure(figsize=(12, 8))
    plt.hist(means, bins=bins, density=True, alpha=0.7, edgecolor="black")

    mu, sigma = np.mean(means), np.std(means, ddof=1)
    x_vals = np.linspace(np.percentile(means, 0.1), np.percentile(means, 99.9), 300)
    plt.plot(x_vals, stats.norm.pdf(x_vals, mu, sigma), 'r--', lw=2, alpha=0.6, label="Gaussiana di riferimento")

    # percentili 2.5 e 97.5
    plt.axvline(np.percentile(means, 2.5), color='orange', linestyle='--', lw=2, label='2.5% percentile')
    plt.axvline(np.percentile(means, 97.5), color='orange', linestyle='--', lw=2, label='97.5% percentile')

    plt.title(f"Bootstrap means — {title}")
    plt.xlabel("bootstrap means")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------------------  
# Processi con memoria  
# ------------------------------  
def ornstein_uhlenbeck(n, theta=0.5, mu=0.0, x0=0.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x = np.zeros(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = x[t-1] + theta*(mu-x[t-1]) + rng.normal()
    return x

def fgn(n, hurst, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    gamma = lambda k: 0.5*((abs(k+1)**(2*hurst) - 2*abs(k)**(2*hurst) + abs(k-1)**(2*hurst)))
    cov = np.array([gamma(k) for k in range(n)])
    L = cholesky(toeplitz(cov), lower=True)
    return L @ rng.normal(size=n)

experiments = []

# OU
for theta in [0.01, 1.2, 2]:
    x = ornstein_uhlenbeck(n, theta=theta, rng=rng)
    m = bootstrap_means(x, B, rng)
    experiments.append((m, summarize_bootstrap(m, f"OU theta={theta}", original=x)))

# fGn
for H in [0.5, 0.75, 0.95]:
    x = fgn(n, H, rng)
    m = bootstrap_means(x, B, rng)
    experiments.append((m, summarize_bootstrap(m, f"fGn H={H}", original=x)))

# ------------------------------  
# Esegui esperimenti  
# ------------------------------  
for means, summary in experiments:
    print_summary(summary)
    plot_means(summary["label"], means)
