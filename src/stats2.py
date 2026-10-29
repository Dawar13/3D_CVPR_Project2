"""EXP2 statistics (pre-registration P3-P6, P9): bootstrap CIs, paired Wilcoxon, Holm, cluster-bootstrap slope."""
import numpy as np
from scipy.stats import wilcoxon

N_BOOT = 1000


def boot_median_ci(x, n_boot=N_BOOT, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = np.median(rng.choice(x, (n_boot, len(x))), 1)
    return float(np.median(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_ratio_ci(num, den, n_boot=N_BOOT, seed=0, paired=True):
    """Median(num) / median(den) with a bootstrap CI; paired resamples the same windows for both."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    r = np.median(num) / np.median(den)
    if paired:
        idx = rng.integers(0, len(num), (n_boot, len(num)))
        b = np.median(num[idx], 1) / np.median(den[idx], 1)
    else:
        b = (np.median(rng.choice(num, (n_boot, len(num))), 1) / np.median(rng.choice(den, (n_boot, len(den))), 1))
    return float(r), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def recovery_ci(v2, m, v4, n_boot=N_BOOT, seed=0):
    """Share of the oracle gain a method recovers: (med V2 - med M) / (med V2 - med V4), paired by window."""
    v2, m, v4 = (np.asarray(a, float) for a in (v2, m, v4))
    f = lambda a, b, c: (np.median(a) - np.median(b)) / (np.median(a) - np.median(c))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v2), (n_boot, len(v2)))
    b = np.array([f(v2[i], m[i], v4[i]) for i in idx])
    return float(f(v2, m, v4)), float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))


def paired_wilcoxon(a, b, alternative="two-sided"):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.allclose(a[ok], b[ok]):
        return np.nan
    return float(wilcoxon(a[ok], b[ok], alternative=alternative).pvalue)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as the input; NaN stays NaN)."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        out[i] = running
    return out


def slope_cluster_boot(x, y, clusters, n_boot=N_BOOT, seed=0, log_y=True):
    """OLS slope of (log) y on x with a two-stage cluster bootstrap (resample clusters, then windows within).
    Returns slope, CI low, CI high, one-sided bootstrap p for slope <= 0 (pre-registration P3)."""
    x, y, clusters = np.asarray(x, float), np.asarray(y, float), np.asarray(clusters)
    if log_y:
        y = np.log(y)
    slope = lambda xx, yy: np.polyfit(xx, yy, 1)[0]
    s0 = slope(x, y)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(clusters == c) for c in np.unique(clusters)]
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([rng.choice(groups[g], len(groups[g])) for g in pick])
        bs.append(slope(x[idx], y[idx]))
    bs = np.array(bs)
    return float(s0), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float((bs <= 0).mean())
