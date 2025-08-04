import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import load_occurrences, load_refs_from_stm, is_match

def get_total_duration_from_stm(stm_path):
    total = 0.0
    with open(stm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                start = float(parts[3])
                end = float(parts[4])
                total += (end - start)
    return total

def evaluate_twv(results, refs, T, beta=20):
    from collections import defaultdict
    grouped_refs = defaultdict(list)
    for ref in refs:
        grouped_refs[ref[0]].append(ref)

    total = 0
    for k, rk in grouped_refs.items():
        hk = [h for h in results if h[0] == k]
        nc = len(rk)
        nm = sum(1 for r in rk if not any(is_match(r[2], r[3], h[2], h[3]) and h[1] == r[1] for h in hk))
        nf = sum(1 for h in hk if not any(is_match(h[2], h[3], r[2], r[3]) and h[1] == r[1] for r in rk))
        nt = T - nc
        pmiss = nm / nc if nc else 1.0
        pfa = nf / nt if nt else 0.0
        total += 1 - (pmiss + beta * pfa)
    return total / len(grouped_refs) if grouped_refs else 0.0

def plot_twv_curve(results, refs, T, output_path="plots/twv_curve.png"):
    thresholds = [round(t, 2) for t in list(i / 100 for i in range(0, 101, 2))]
    twv_scores = []

    for t in thresholds:
        filtered = [r for r in results if r[4] >= t]
        score = evaluate_twv(filtered, refs, T)
        twv_scores.append(score)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure()
    plt.plot(thresholds, twv_scores, marker='o')
    plt.xlabel("Threshold")
    plt.ylabel("TWV")
    plt.title("TWV Across Thresholds")
    plt.grid(True)
    plt.savefig(output_path)
    print(f"[INFO] TWV plot saved to {output_path}")

    # Report ATWV and MTWV
    atwv = evaluate_twv(results, refs, T)
    mtwv = max(twv_scores)
    best_thresh = thresholds[twv_scores.index(mtwv)]

    print(f"ATWV: {atwv:.4f}")
    print(f"MTWV: {mtwv:.4f} at threshold {best_thresh:.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python twv.py <keywords> <occurrences> <stm>")
        sys.exit(1)

    _, kw_path, occ_path, stm_path = sys.argv
    with open(kw_path) as f:
        keywords = set(line.strip().lower() for line in f if line.strip())

    from utils import load_occurrences, load_refs_from_stm

    results = load_occurrences(occ_path)
    refs = load_refs_from_stm(stm_path, keywords)
    total_duration = get_total_duration_from_stm(stm_path)

    plot_twv_curve(results, refs, total_duration)