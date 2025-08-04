# apply.py: Apply saved weights to evaluation data
import sys
import numpy as np
import glob

def load_ngram_prob_files(file_paths, min_prob=1e-9):
    all_probs = []
    for path in file_paths:
        with open(path) as f:
            values = [max(float(line.strip()), min_prob) for line in f if line.strip()]
            all_probs.append(np.array(values))
    return all_probs

def load_weights(path):
    with open(path) as f:
        return np.array([float(line.strip()) for line in f if line.strip()])

def interpolate(probs, weights):
    return np.sum([w * p for w, p in zip(weights, probs)], axis=0)

def compute_perplexity(probs):
    return np.exp(-np.mean(np.log(np.clip(probs, 1e-12, 1.0))))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python apply.py <eval_glob> <weights_file>")
        sys.exit(1)

    eval_glob = sys.argv[1]
    weights_file = sys.argv[2]

    eval_probs = load_ngram_prob_files(glob.glob(eval_glob))
    weights = load_weights(weights_file)

    interp = interpolate(eval_probs, weights)
    ppl = compute_perplexity(interp)

    print(f"✅ Evaluation perplexity: {ppl:.4f}")
