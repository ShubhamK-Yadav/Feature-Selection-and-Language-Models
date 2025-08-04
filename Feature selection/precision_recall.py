import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, auc
from utils import load_occurrences, load_refs_from_stm, is_match

def evaluate_pr_auc(results, refs):
    from collections import defaultdict
    ref_index = defaultdict(list)
    for ref in refs:
        ref_index[ref[1]].append(ref)

    y_true, y_scores = [], []
    matched_refs = set()

    for r in results:
        file_refs = ref_index[r[1]]
        matched = False
        for ref in file_refs:
            if ref[0] == r[0] and is_match(r[2], r[3], ref[2], ref[3]):
                matched = True
                matched_refs.add((ref[0], ref[1], ref[2], ref[3]))
                break
        y_true.append(int(matched))
        y_scores.append(r[4])

    print(f"[INFO] Matched: {len(matched_refs)} | Missed: {len(refs) - len(matched_refs)}")

    if not any(y_true):
        return [], [], 0.0
    p, r, _ = precision_recall_curve(y_true, y_scores)
    return p, r, auc(r, p)

def plot_precision_recall(precision, recall, auc_score, output_path="plots/pr_curve.png"):
    os.makedirs("plots", exist_ok=True)
    plt.figure()
    plt.plot(recall, precision, marker='.', label=f"AUC = {auc_score:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] PR curve saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python precision_recall.py <keywords> <occurrences> <stm>")
        sys.exit(1)

    _, kw_path, occ_path, stm_path = sys.argv
    with open(kw_path) as f:
        keywords = set(line.strip().lower() for line in f if line.strip())

    results = load_occurrences(occ_path)
    refs = load_refs_from_stm(stm_path, keywords)
    p, r, auc_score = evaluate_pr_auc(results, refs)

    print(f"[RESULT] AUC: {auc_score:.4f}")
    if len(p) > 0 and len(r) > 0:
        plot_precision_recall(p, r, auc_score)