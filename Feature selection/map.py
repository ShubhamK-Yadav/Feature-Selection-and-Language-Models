import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve
from utils import load_occurrences, load_refs_from_stm, is_match

def plot_precision_recall_curve(y_true, y_scores, mAP_score, output_path="plots/map_precision_recall.png"):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, label=f"mAP = {mAP_score:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (mAP)")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"[INFO] Precision-Recall plot saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python map.py <keywords> <occurrences> <stm>")
        sys.exit(1)

    _, kw_path, occ_path, stm_path = sys.argv
    with open(kw_path) as f:
        keywords = set(line.strip().lower() for line in f if line.strip())

    results = load_occurrences(occ_path)
    refs = load_refs_from_stm(stm_path, keywords)

    from collections import defaultdict
    ref_index = defaultdict(list)
    for ref in refs:
        ref_index[ref[1]].append(ref)

    y_true, y_scores = [], []
    for r in results:
        matched = any(
            ref[0] == r[0] and is_match(r[2], r[3], ref[2], ref[3])
            for ref in ref_index[r[1]]
        )
        y_true.append(int(matched))
        y_scores.append(r[4])

    mAP = average_precision_score(y_true, y_scores) if any(y_true) else 0.0
    print(f"mAP: {mAP:.4f}")
    if any(y_true):
        plot_precision_recall_curve(y_true, y_scores, mAP)
