import sys
import numpy as np
import glob

def load_probs(file_paths, min_prob=1e-9):
    probs = []
    for path in file_paths:
        with open(path) as f:
            values = [max(float(line.strip()), min_prob) for line in f if line.strip()]
            probs.append(np.array(values))
    return probs

def load_weights(path):
    with open(path) as f:
        return np.array([float(line.strip()) for line in f if line.strip()])

def interpolate(probs, weights):
    return np.sum([w * p for w, p in zip(weights, probs)], axis=0)

def compute_perplexity(interp_probs):
    return np.exp(-np.mean(np.log(np.clip(interp_probs, 1e-12, 1.0))))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python apply.py <weights_file> <eval_glob>")
        sys.exit(1)

    weight_path = sys.argv[1]
    eval_glob = sys.argv[2]

    eval_probs = load_probs(glob.glob(eval_glob))
    weights = load_weights(weight_path)

    interp_probs = interpolate(eval_probs, weights)
    ppl = compute_perplexity(interp_probs)

    print(f"Eval perplexity: {ppl:.4f}")
