import sys
import numpy as np
import glob

def load_ngram_prob_files(file_paths, min_prob=1e-9):
    """
    Loads n-gram probability files with additive smoothing.
    """
    all_probs = []
    for path in file_paths:
        with open(path, 'r') as f:
            values = [max(float(line.strip()), min_prob) for line in f if line.strip()]
            all_probs.append(np.array(values))
    return all_probs

def load_hybrid_weights(weight_path, L):
    """
    Loads global and group weights from the hybrid EM weight file.
    """
    with open(weight_path) as f:
        weights = [float(line.strip()) for line in f if line.strip()]
    global_weights = np.array(weights[:L])
    group_weights = np.array(weights[L:]).reshape(-1, L)
    return global_weights, group_weights

def assign_groups_by_average_probability(probs, num_groups=3):
    """
    Assigns group IDs to each word position based on average probability.
    """
    avg_probs = np.mean(np.array(probs), axis=0)
    thresholds = np.quantile(avg_probs, np.linspace(0, 1, num_groups + 1)[1:-1])
    return np.digitize(avg_probs, thresholds)

def interpolate_probs(probs, weights):
    """
    Interpolates probabilities using the provided weight vector.
    """
    return np.sum([w * p for w, p in zip(weights, probs)], axis=0)

def compute_perplexity(log_probs):
    """
    Computes perplexity from log probabilities.
    """
    return np.exp(-np.mean(log_probs))

def apply_hybrid(eval_probs, global_weights, group_weights, gamma=0.5, num_groups=3):
    """
    Applies the hybrid EM interpolation to compute perplexity on eval data.
    """
    T = len(eval_probs[0])
    group_ids = assign_groups_by_average_probability(eval_probs, num_groups)
    log_probs = np.zeros(T)

    for g in range(num_groups):
        mask = (group_ids == g)
        mixed_weights = gamma * global_weights + (1 - gamma) * group_weights[g]
        interp_probs = interpolate_probs(eval_probs, mixed_weights)[mask]
        log_probs[mask] = np.log(np.clip(interp_probs, 1e-12, 1.0))

    return compute_perplexity(log_probs)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python apply.py <eval_glob> <weight_file>")
        sys.exit(1)

    eval_glob = sys.argv[1]
    weight_file = sys.argv[2]

    eval_files = glob.glob(eval_glob)
    eval_probs = load_ngram_prob_files(eval_files)

    L = len(eval_probs)
    global_w, group_w = load_hybrid_weights(weight_file, L)

    ppl = apply_hybrid(eval_probs, global_w, group_w)

    print(f"Final evaluation perplexity (hybrid EM): {ppl:.4f}")
