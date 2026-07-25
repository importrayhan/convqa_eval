#!/usr/bin/env python3
"""
significance.py — Statistical significance & model agreement analysis.

Builds the canonical per-query metric matrix (rows=queries, cols=systems)
and computes:

  1. Pairwise significance: paired t-test, Wilcoxon signed-rank,
     paired randomization (Smucker et al. CIKM 2007), paired bootstrap
     (Sakai SIGIR 2006)
  2. Omnibus test: repeated-measures ANOVA (Friedman non-parametric)
  3. Inter-system agreement: Cohen's kappa (pairwise), Fleiss' kappa
     (multi-system), Kendall's W (concordance)
  4. Normalized NLL efficiency: 1 - NLL/H(label), a lower bound on
     the normalized mutual information the model extracts from data.
     Higher = model is more reliable.

Usage:
  python scripts/significance.py \
      --predictions pred_A.json pred_B.json pred_C.json \
      --gold benchmarks/pacific/data/test.json \
      --benchmark pacific --track cnp
"""

import json, argparse, logging, glob, sys
from pathlib import Path
from itertools import combinations

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ══════════════════════════════════════════════════════════════════════════════
# Per-query metric extraction
# ══════════════════════════════════════════════════════════════════════════════
def extract_per_query(submission, gold, track="cnp"):
    """Extract per-query metric and confidence for a submission."""
    preds = submission.get("predictions", [])
    per_q = {}
    confs = {}
    for conv in preds:
        cid = conv.get("conv_id", "")
        for turn in conv.get("turns", []):
            key = (cid, turn["turn_idx"])
            if key not in gold:
                continue
            g = gold[key]["label"]
            p = turn["pred"]
            conf = turn.get("confidence", None)

            if track == "cnp":
                per_q[key] = 1.0 if p == g else 0.0
            elif track == "defense":
                flipped = turn.get("pred_clean", p) != turn.get("pred_adv", p)
                flagged = turn.get("defense_flag", False)
                per_q[key] = (1.0 if flagged else 0.0) if flipped else \
                             (1.0 if not flagged else 0.0)
            confs[key] = conf
    return per_q, confs


def build_matrix(submissions, gold, track):
    """Build (n_queries x n_systems) metric matrix + label/confidence arrays."""
    all_pq, all_conf = {}, {}
    for name, sub in submissions.items():
        pq, cf = extract_per_query(sub, gold, track)
        all_pq[name] = pq
        all_conf[name] = cf

    common = None
    for pq in all_pq.values():
        common = set(pq) if common is None else common & set(pq)
    common = sorted(common)
    names = sorted(submissions.keys())

    M = np.zeros((len(common), len(names)))
    for j, n in enumerate(names):
        for i, k in enumerate(common):
            M[i, j] = all_pq[n].get(k, 0.0)

    # Gold labels and per-system confidences for NLL
    gold_labels = np.array([gold[k]["label"] for k in common])
    conf_matrix = {}
    for j, n in enumerate(names):
        cf = []
        for k in common:
            c = all_conf[n].get(k, None)
            cf.append(c if c is not None else (0.5 if M[i, j] == 0 else 0.5))
        conf_matrix[n] = np.array(cf)

    # Also extract raw predictions for agreement analysis
    pred_matrix = np.zeros((len(common), len(names)), dtype=int)
    for j, n in enumerate(names):
        sub = submissions[n]
        pq_raw = {}
        for conv in sub.get("predictions", []):
            for turn in conv.get("turns", []):
                pq_raw[(conv.get("conv_id", ""), turn["turn_idx"])] = turn["pred"]
        for i, k in enumerate(common):
            pred_matrix[i, j] = pq_raw.get(k, 0)

    return common, names, M, gold_labels, conf_matrix, pred_matrix


# ══════════════════════════════════════════════════════════════════════════════
# Pairwise significance tests
# ══════════════════════════════════════════════════════════════════════════════
def paired_randomization(a, b, n_iter=10000, seed=42):
    rng = np.random.RandomState(seed)
    obs = abs(np.mean(a) - np.mean(b))
    n = len(a)
    count = 0
    for _ in range(n_iter):
        swaps = rng.randint(0, 2, size=n).astype(bool)
        d = abs(np.mean(np.where(swaps, b, a)) - np.mean(np.where(swaps, a, b)))
        if d >= obs:
            count += 1
    return count / n_iter


def paired_bootstrap(a, b, n_iter=10000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(a)
    obs = np.mean(a) - np.mean(b)
    count = 0
    for _ in range(n_iter):
        idx = rng.randint(0, n, size=n)
        d = np.mean(a[idx]) - np.mean(b[idx])
        if (obs >= 0 and d <= 0) or (obs < 0 and d >= 0):
            count += 1
    return min(count / n_iter * 2, 1.0)


def all_pairwise_tests(names, M, n_iter=10000):
    """Run all four pairwise tests between every system pair."""
    from scipy.stats import ttest_rel, wilcoxon
    results = {}
    for i, j in combinations(range(len(names)), 2):
        a, b = M[:, i], M[:, j]
        diff = float(np.mean(a) - np.mean(b))

        _, p_t = ttest_rel(a, b)
        try:
            _, p_w = wilcoxon(a, b, alternative="two-sided")
        except ValueError:
            p_w = 1.0  # all differences zero
        p_r = paired_randomization(a, b, n_iter)
        p_b = paired_bootstrap(a, b, n_iter)

        results[(names[i], names[j])] = {
            "mean_diff": diff,
            "p_ttest": float(p_t),
            "p_wilcoxon": float(p_w),
            "p_randomization": float(p_r),
            "p_bootstrap": float(p_b),
        }
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Omnibus test
# ══════════════════════════════════════════════════════════════════════════════
def omnibus_test(M):
    """Friedman test: are there significant differences among all systems?"""
    from scipy.stats import friedmanchisquare
    if M.shape[1] < 3:
        return {"chi2": 0, "p_value": 1.0, "note": "need >=3 systems"}
    cols = [M[:, j] for j in range(M.shape[1])]
    try:
        chi2, p = friedmanchisquare(*cols)
        return {"chi2": float(chi2), "p_value": float(p)}
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# Agreement analysis
# ══════════════════════════════════════════════════════════════════════════════
def cohens_kappa(pred_a, pred_b):
    """Cohen's kappa between two systems' predictions."""
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(pred_a, pred_b)


def fleiss_kappa(pred_matrix, n_classes):
    """Fleiss' kappa for multi-rater agreement."""
    n, k = pred_matrix.shape  # n=queries, k=systems
    # Build category count matrix
    C = np.zeros((n, n_classes))
    for i in range(n):
        for j in range(k):
            C[i, pred_matrix[i, j]] += 1
    p_j = C.sum(axis=0) / (n * k)
    P_i = (np.sum(C ** 2, axis=1) - k) / (k * (k - 1))
    P_bar = np.mean(P_i)
    P_e = np.sum(p_j ** 2)
    if abs(1 - P_e) < 1e-10:
        return 1.0 if P_bar == 1.0 else 0.0
    return float((P_bar - P_e) / (1 - P_e))


def kendalls_w(M):
    """Kendall's W concordance coefficient across systems."""
    from scipy.stats import rankdata
    n, k = M.shape
    ranks = np.apply_along_axis(rankdata, 0, M)
    rank_sums = ranks.sum(axis=1)
    mean_rank_sum = np.mean(rank_sums)
    ss = np.sum((rank_sums - mean_rank_sum) ** 2)
    W = 12 * ss / (k ** 2 * (n ** 3 - n))
    return float(np.clip(W, 0, 1))


def agreement_analysis(names, pred_matrix, n_classes):
    """Full agreement analysis: pairwise kappa + Fleiss + Kendall."""
    pairwise_kappa = {}
    for i, j in combinations(range(len(names)), 2):
        k = cohens_kappa(pred_matrix[:, i], pred_matrix[:, j])
        pairwise_kappa[(names[i], names[j])] = float(k)

    fk = fleiss_kappa(pred_matrix, n_classes)
    return {"pairwise_cohen_kappa": pairwise_kappa, "fleiss_kappa": fk}


# ══════════════════════════════════════════════════════════════════════════════
# Normalized NLL / mutual information efficiency
# ══════════════════════════════════════════════════════════════════════════════
def normalized_nll_efficiency(gold_labels, pred_matrix, conf_matrix,
                               names, n_classes):
    """
    Compute 1 - NLL/H(label) per system.

    NLL = -(1/N) Σ log P(y_true | x)
    H(label) = prior label entropy

    The quantity 1 - NLL/H(label) is a lower bound on the normalized
    mutual information between features and labels.  Higher = model
    extracts more information from data = more reliable.

    If confidence scores are not available, we estimate P(y_true | x)
    from per-query accuracy (1.0 for correct, smoothed to 0.01 for
    incorrect, which gives a pessimistic NLL).
    """
    N = len(gold_labels)
    # Prior label entropy
    counts = np.bincount(gold_labels, minlength=n_classes)
    probs = counts / N
    probs = probs[probs > 0]
    H_label = float(-np.sum(probs * np.log(probs)))

    if H_label < 1e-10:
        return {n: {"nll": 0, "h_label": 0, "efficiency": 0} for n in names}

    results = {}
    for j, name in enumerate(names):
        # P(y_true | x) from confidence if available, else from accuracy
        conf = conf_matrix.get(name)
        nll_terms = []
        for i in range(N):
            correct = (pred_matrix[i, j] == gold_labels[i])
            if conf is not None and conf[i] is not None and conf[i] > 0:
                # If prediction correct, P(y_true) = confidence
                # If prediction wrong, P(y_true) = 1 - confidence (approx)
                p_true = float(conf[i]) if correct else max(1 - float(conf[i]), 1e-6)
            else:
                p_true = 0.99 if correct else 0.01  # pessimistic estimate
            p_true = np.clip(p_true, 1e-6, 1 - 1e-6)
            nll_terms.append(-np.log(p_true))

        nll = float(np.mean(nll_terms))
        efficiency = 1.0 - nll / H_label

        results[name] = {
            "nll": round(nll, 4),
            "h_label": round(H_label, 4),
            "efficiency": round(efficiency, 4),
            "nll_normalized": round(nll / H_label, 4),
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════════════
def plot_analysis(names, M, pairwise, agreement, nll_results,
                  friedman, track, benchmark, output_dir):
    """
    4-panel poster figure:
      (a) Per-system boxplot with means
      (b) Pairwise significance heatmap (randomization p-values + stars)
      (c) Agreement matrix (Cohen's kappa heatmap)
      (d) NLL efficiency bar chart
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.labelsize": 12, "axes.titlesize": 13,
    })

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    n = len(names)

    # ── (a) Boxplot ──────────────────────────────────────────────────
    ax = axes[0, 0]
    means = [np.mean(M[:, j]) for j in range(n)]
    order = np.argsort(means)[::-1]
    colors = ["#E3F2FD", "#E8F5E9", "#FFF3E0", "#FCE4EC",
              "#F3E5F5", "#E0F7FA", "#FFF9C4", "#EFEBE9"]

    bp = ax.boxplot(
        [M[:, j] for j in order],
        labels=[names[j] for j in order],
        patch_artist=True, widths=0.6,
        medianprops=dict(color="#C62828", linewidth=2),
    )
    for k, (patch, idx) in enumerate(zip(bp["boxes"], order)):
        patch.set_facecolor(colors[idx % len(colors)])
        ax.plot(k + 1, means[idx], "D", color="#1565C0", ms=7, zorder=5)

    metric_name = "Per-query Accuracy" if track == "cnp" else "Defense Success"
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title("(a) Per-System Distribution", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, alpha=0.15, axis="y")

    # Friedman annotation
    if "p_value" in friedman:
        fp = friedman["p_value"]
        stars = "***" if fp < 0.001 else "**" if fp < 0.01 else "*" if fp < 0.05 else "ns"
        ax.text(0.98, 0.02, f"Friedman p={fp:.4f} {stars}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # ── (b) Significance heatmap ─────────────────────────────────────
    ax = axes[0, 1]
    p_mat = np.ones((n, n))
    for (a, b), res in pairwise.items():
        i, j = names.index(a), names.index(b)
        p_mat[i, j] = res["p_randomization"]
        p_mat[j, i] = res["p_randomization"]

    im = ax.imshow(p_mat, cmap="RdYlGn", vmin=0, vmax=0.1)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=10)
            else:
                p = p_mat[i, j]
                s = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                c = "white" if p < 0.05 else "black"
                ax.text(j, i, f"{p:.3f}\n{s}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color=c)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, fontsize=9, rotation=30, ha="right")
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title("(b) Pairwise Significance (randomization)", fontsize=13,
                 fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.75, label="p-value")

    # ── (c) Agreement heatmap (Cohen's kappa) ────────────────────────
    ax = axes[1, 0]
    kappa_mat = np.ones((n, n))
    pk = agreement.get("pairwise_cohen_kappa", {})
    for (a, b), k_val in pk.items():
        i, j = names.index(a), names.index(b)
        kappa_mat[i, j] = k_val
        kappa_mat[j, i] = k_val

    im2 = ax.imshow(kappa_mat, cmap="Blues", vmin=-0.2, vmax=1.0)
    for i in range(n):
        for j in range(n):
            val = kappa_mat[i, j]
            c = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=c)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, fontsize=9, rotation=30, ha="right")
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title(f"(c) Agreement (Cohen's \u03ba, Fleiss'={agreement.get('fleiss_kappa',0):.3f})",
                 fontsize=13, fontweight="bold")
    fig.colorbar(im2, ax=ax, shrink=0.75, label="Cohen's \u03ba")

    # ── (d) NLL efficiency ───────────────────────────────────────────
    ax = axes[1, 1]
    eff_names = sorted(nll_results.keys(),
                       key=lambda x: nll_results[x]["efficiency"], reverse=True)
    eff_vals = [nll_results[n]["efficiency"] for n in eff_names]
    nll_vals = [nll_results[n]["nll"] for n in eff_names]

    x_pos = np.arange(len(eff_names))
    bars = ax.bar(x_pos, eff_vals, color="#1565C0", alpha=0.85,
                  edgecolor="#0D47A1", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(eff_names, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("1 \u2212 NLL/H(label)", fontsize=12)
    ax.set_title("(d) Normalized Information Efficiency", fontsize=13,
                 fontweight="bold")
    ax.grid(True, alpha=0.15, axis="y")

    for bar, val, nll in zip(bars, eff_vals, nll_vals):
        ax.text(bar.get_x() + bar.get_width()/2, max(val, 0) + 0.02,
                f"{val:.3f}\n(NLL={nll:.2f})", ha="center", fontsize=8)

    h_label = list(nll_results.values())[0]["h_label"]
    ax.text(0.98, 0.02, f"H(label)={h_label:.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    track_label = "Clarification" if track == "cnp" else "Defense"
    fig.suptitle(f"Statistical Analysis — {benchmark} ({track_label})",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()

    p = out / f"significance_{benchmark}_{track}.png"
    fig.savefig(p, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"Plot: {p}")
    return str(p)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Statistical significance & agreement analysis")
    ap.add_argument("--predictions", nargs="+", required=True)
    ap.add_argument("--gold", type=str, required=True)
    ap.add_argument("--benchmark", type=str, default="pacific")
    ap.add_argument("--track", default="cnp", choices=["cnp", "defense"])
    ap.add_argument("--n_iter", type=int, default=10000)
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--per_turn", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default="llm_results")
    args = ap.parse_args()

    pred_paths = []
    for p in args.predictions:
        pred_paths.extend(glob.glob(p))

    from convqa_eval.data.final_loader import BENCHMARK_REGISTRY
    per_turn = (args.per_turn.lower() == "true") if args.per_turn else \
               BENCHMARK_REGISTRY.get(args.benchmark, {}).get("per_turn_default", True)

    from eval_preditions import load_gold_labels
    gold = load_gold_labels(args.gold, args.benchmark, per_turn, args.num_classes)
    log.info(f"Gold: {len(gold)} turns")

    submissions = {}
    for path in pred_paths:
        sub = json.load(open(path))
        name = sub.get("metadata", {}).get("method",
               sub.get("metadata", {}).get("team", Path(path).stem))
        if name in submissions:
            name = f"{name}_{Path(path).stem[-4:]}"
        submissions[name] = sub

    keys, names, M, gold_labels, conf_matrix, pred_matrix = \
        build_matrix(submissions, gold, args.track)
    log.info(f"Matrix: {M.shape[0]} queries x {M.shape[1]} systems")

    # Per-system means
    log.info(f"\nPer-system mean ({args.track}):")
    for j, n in enumerate(names):
        log.info(f"  {n:20s} = {np.mean(M[:,j]):.4f} (±{np.std(M[:,j]):.4f})")

    # Pairwise tests
    log.info(f"\nPairwise tests (n_iter={args.n_iter}):")
    pw = all_pairwise_tests(names, M, args.n_iter)
    for (a, b), r in sorted(pw.items()):
        s = "***" if r["p_randomization"] < 0.001 else \
            "**" if r["p_randomization"] < 0.01 else \
            "*" if r["p_randomization"] < 0.05 else "ns"
        log.info(f"  {a:16s} vs {b:16s}  Δ={r['mean_diff']:+.4f}  "
                 f"t={r['p_ttest']:.4f}  W={r['p_wilcoxon']:.4f}  "
                 f"R={r['p_randomization']:.4f}  B={r['p_bootstrap']:.4f}  {s}")

    # Omnibus
    friedman = omnibus_test(M)
    log.info(f"\nFriedman test: {friedman}")

    # Agreement
    agree = agreement_analysis(names, pred_matrix, args.num_classes)
    log.info(f"\nFleiss' kappa: {agree['fleiss_kappa']:.4f}")
    for (a, b), k in sorted(agree["pairwise_cohen_kappa"].items()):
        log.info(f"  Cohen's kappa({a}, {b}) = {k:.4f}")

    # Kendall's W
    W = kendalls_w(M)
    log.info(f"Kendall's W = {W:.4f}")

    # NLL efficiency
    nll = normalized_nll_efficiency(gold_labels, pred_matrix,
                                     conf_matrix, names, args.num_classes)
    log.info(f"\nNormalized NLL Efficiency (1 - NLL/H(label)):")
    for n in sorted(nll, key=lambda x: nll[x]["efficiency"], reverse=True):
        log.info(f"  {n:20s}  eff={nll[n]['efficiency']:.4f}  "
                 f"NLL={nll[n]['nll']:.4f}  H={nll[n]['h_label']:.4f}")

    # Plot
    plot_analysis(names, M, pw, agree, nll, friedman,
                  args.track, args.benchmark, args.output_dir)

    # Save
    out = Path(args.output_dir) / f"significance_{args.benchmark}_{args.track}.json"
    with open(out, "w") as f:
        json.dump({
            "benchmark": args.benchmark, "track": args.track,
            "n_queries": M.shape[0], "n_systems": M.shape[1],
            "systems": {n: {"mean": float(np.mean(M[:,j])),
                            "std": float(np.std(M[:,j]))}
                        for j, n in enumerate(names)},
            "pairwise": {f"{a}_vs_{b}": r for (a,b), r in pw.items()},
            "friedman": friedman,
            "agreement": {
                "fleiss_kappa": agree["fleiss_kappa"],
                "kendalls_w": W,
                "pairwise_kappa": {f"{a}_vs_{b}": k
                    for (a,b), k in agree["pairwise_cohen_kappa"].items()},
            },
            "nll_efficiency": nll,
        }, f, indent=2)
    log.info(f"Results: {out}")


if __name__ == "__main__":
    main()
