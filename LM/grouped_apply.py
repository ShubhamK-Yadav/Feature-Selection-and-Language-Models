import sys
import numpy as np
import glob

def load_ngram_prob_files_additive_with_threshold(file_paths, min_prob=1e-9):
    all_probs = []
    for path in file_paths:
        with open(path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            values = [max(float(line), min_prob) for line in lines]
            all_probs.append(np.array(values))
    return all_probs

def load_group_weights(weight_file, L):
    with open(weight_file, 'r') as f:
        flat_weights = [float(line.strip()) for line in f if line.strip()]
    group_weights = np.array(flat_weights).reshape(-1, L)
    return group_weights

def assign_groups_by_average_probability(probs, num_groups=2):
    avg_probs = np.mean(np.array(probs), axis=0)
    thresholds = np.quantile(avg_probs, np.linspace(0, 1, num_groups + 1)[1:-1])
    return np.digitize(avg_probs, thresholds)

def interpolate_probs(probs, weights):
    return np.sum([w * p for w, p in zip(weights, probs)], axis=0)

def compute_perplexity(interpolated_probs):
    log_probs = np.log(np.clip(interpolated_probs, 1e-12, 1.0))
    return np.exp(-np.mean(log_probs))

def apply_group_weights(eval_probs, group_weights, num_groups=2):
    group_ids = assign_groups_by_average_probability(eval_probs, num_groups)
    T = len(eval_probs[0])
    log_probs = np.zeros(T)

    for g in range(num_groups):
        mask = (group_ids == g)
        interpolated = interpolate_probs(eval_probs, group_weights[g])[mask]
        log_probs[mask] = np.log(np.clip(interpolated, 1e-12, 1.0))

    return np.exp(-np.mean(log_probs))

if __name__ == "__main__":
    if '*' in sys.argv[1]:
        eval_files = glob.glob(sys.argv[1])
        weight_file = sys.argv[2]
    else:
        eval_files = sys.argv[1:-1]
        weight_file = sys.argv[-1]

    eval_probs = load_ngram_prob_files_additive_with_threshold(eval_files)
    L = len(eval_probs)

    group_weights = load_group_weights(weight_file, L)
    num_groups = group_weights.shape[0]

    perplexity = apply_group_weights(eval_probs, group_weights, num_groups=num_groups)

    print(f"Final evaluation perplexity (group-based): {perplexity:.4f}")
