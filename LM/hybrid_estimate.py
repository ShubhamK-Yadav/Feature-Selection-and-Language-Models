# estimate.py: Hybrid EM using only dev data (Assignment-compliant)
import sys
import numpy as np
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def init_weights(strategy="uniform", L=10, top_indices=[0, 5], top_value=0.4):
    if strategy == "uniform":
        return np.full(L, 1.0 / L)
    elif strategy == "random":
        weights = np.random.rand(L)
        return weights / np.sum(weights)
    elif strategy == "top-heavy":
        weights = np.full(L, (1.0 - top_value) / (L - len(top_indices)))
        for idx in top_indices:
            weights[idx] = top_value / len(top_indices)
        return weights / np.sum(weights)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def load_ngram_prob_files_additive_with_threshold(file_paths, min_prob=1e-9):
    all_probs = []
    for path in file_paths:
        with open(path, 'r') as f:
            values = [max(float(line.strip()), min_prob) for line in f if line.strip()]
            all_probs.append(np.array(values))
    return all_probs

def interpolate_probs(probs, weights):
    return np.sum([w * p for w, p in zip(weights, probs)], axis=0)

def compute_perplexity(log_probs):
    return np.exp(-np.mean(log_probs))

def assign_groups_by_average_probability(probs, num_groups=3):
    avg_probs = np.mean(np.array(probs), axis=0)
    thresholds = np.quantile(avg_probs, np.linspace(0, 1, num_groups + 1)[1:-1])
    return np.digitize(avg_probs, thresholds)

def hybrid_em(dev_probs, beta=1.0, max_iter=100, tol=1e-4, alpha=0.001, num_groups=3, gamma=0.5):
    L = len(dev_probs)
    T = len(dev_probs[0])
    group_ids = assign_groups_by_average_probability(dev_probs, num_groups)
    global_weights = init_weights(strategy="top-heavy", L=L, top_indices=[0, 5], top_value=0.6)
    group_weights = np.full((num_groups, L), 1.0 / L)
    dev_ppls = []

    best_dev_ppl = float("inf")
    best_global_weights = global_weights.copy()
    best_group_weights = group_weights.copy()
    best_index = 0

    for i in range(max_iter):
        dev_log_probs = np.zeros(T)

        for g in range(num_groups):
            mask = (group_ids == g)
            gw = group_weights[g]
            mix_weights = gamma * global_weights + (1 - gamma) * gw
            dev_interp = interpolate_probs(dev_probs, mix_weights)[mask]
            dev_log_probs[mask] = np.log(np.clip(dev_interp, 1e-12, 1.0))

        dev_ppl = compute_perplexity(dev_log_probs)
        dev_ppls.append(dev_ppl)

        if dev_ppl < best_dev_ppl:
            best_dev_ppl = dev_ppl
            best_index = i
            best_global_weights = global_weights.copy()
            best_group_weights = group_weights.copy()

        # Update each group
        for g in range(num_groups):
            mask = (group_ids == g)
            posteriors = np.zeros((L, np.sum(mask)))
            for l in range(L):
                posteriors[l] = group_weights[g][l] * (dev_probs[l][mask] ** beta)
            denom = np.sum(posteriors, axis=0)
            posteriors /= denom
            new_weights = np.sum(posteriors, axis=1) / posteriors.shape[1]
            entropy = -np.sum(new_weights * np.log(np.clip(new_weights, 1e-12, 1.0)))
            new_weights += alpha * entropy
            new_weights = np.clip(new_weights, 1e-12, None)
            group_weights[g] = new_weights / np.sum(new_weights)

        # Update global weights
        posteriors = np.zeros((L, T))
        for l in range(L):
            posteriors[l] = global_weights[l] * (dev_probs[l] ** beta)
        denom = np.sum(posteriors, axis=0)
        posteriors /= denom
        new_global = np.sum(posteriors, axis=1) / T
        entropy = -np.sum(new_global * np.log(np.clip(new_global, 1e-12, 1.0)))
        new_global += alpha * entropy
        new_global = np.clip(new_global, 1e-12, None)
        global_weights = new_global / np.sum(new_global)

        if i > 0 and abs(dev_ppls[-1] - dev_ppls[-2]) < tol:
            break

    return best_global_weights, best_group_weights, dev_ppls, best_index

def plot_perplexity(dev_ppls):
    os.makedirs("plots", exist_ok=True)
    plt.plot(dev_ppls, label="Dev")
    plt.xlabel("EM Iteration")
    plt.ylabel("Perplexity")
    plt.title("Hybrid EM Dev Perplexity")
    plt.legend()
    plt.grid(True)
    path = "plots/hybrid_em_perplexity.png"
    plt.savefig(path)
    return path

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python estimate.py <dev_glob> <output_weights_file>")
        sys.exit(1)

    dev_glob = sys.argv[1]
    output_path = sys.argv[2]

    dev_probs = load_ngram_prob_files_additive_with_threshold(glob.glob(dev_glob))

    global_w, group_w, dev_ppls, best_idx = hybrid_em(dev_probs, beta=1.0, alpha=0.0, gamma=0.0)

    with open(output_path, "w") as f:
        for w in global_w:
            f.write(f"{w:.6f}\n")
        for g in group_w:
            for w in g:
                f.write(f"{w:.6f}\n")

    print("\n Weights saved to:", output_path)
    print("Best dev perplexity:", dev_ppls[best_idx])
    print("Iterations:", best_idx + 1)
    print("Perplexity plot saved to:", plot_perplexity(dev_ppls))
