import sys
import numpy as np
import os
import glob
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt

def load_ngram_prob_files_additive_with_threshold(file_paths, min_prob=1e-9):
    """
    Loads n-gram probability files and applies additive smoothing.
    Replaces values < min_prob with min_prob to avoid log(0).
    """
    all_probs = []
    for path in file_paths:
        with open(path, 'r') as f:
            values = [max(float(line.strip()), min_prob) for line in f if line.strip()]
            all_probs.append(np.array(values))
    return all_probs

def init_weights(strategy, L, top_indices=[0, 5], top_value=0.4):
    """
    Initialise interpolation weights based on the selected strategy.
    - 'uniform': equal weights
    - 'random': random values normalised
    - 'biased': favour a single model (e.g. model 5)
    - 'top-heavy': favour a few selected models
    """
    if strategy == "uniform":
        return np.full(L, 1.0 / L)
    elif strategy == "random":
        weights = np.random.rand(L)
        return weights / np.sum(weights)
    elif strategy == "biased":
        weights = np.full(L, (1.0 - top_value) / (L - 1))
        weights[5] = top_value  # assuming model 6 (index 5) is favoured
        return weights / np.sum(weights)
    elif strategy == "top-heavy":
        weights = np.full(L, (1.0 - top_value) / (L - len(top_indices)))
        for idx in top_indices:
            weights[idx] = top_value / len(top_indices)
        return weights / np.sum(weights)
    else:
        raise ValueError(f"Unknown initialisation strategy: {strategy}")

def interpolate_probs(probs, weights):
    """
    Computes the weighted sum of n-gram probabilities.
    probs: list of arrays (each model)
    weights: corresponding weights (λ)
    Returns: interpolated probability vector
    """
    return np.sum([w * p for w, p in zip(weights, probs)], axis=0)

def compute_perplexity(log_probs):
    """
    Computes perplexity from log-probabilities.
    PPL = exp(-average log probability)
    """
    return np.exp(-np.mean(log_probs))

def assign_groups_by_average_probability(probs, num_groups=3):
    """
    Assigns each word position to a group based on average dev probability.
    Returns an array of group IDs (0-based).
    """
    avg_probs = np.mean(np.array(probs), axis=0)
    thresholds = np.quantile(avg_probs, np.linspace(0, 1, num_groups + 1)[1:-1])
    return np.digitize(avg_probs, thresholds)

def em_group_weights(dev_probs, beta=1.0, max_iter=100, tol=1e-4, alpha=0.0, num_groups=2):
    """
    Estimates group-based interpolation weights using EM.
    Returns best group weights, perplexities per iteration, and best index.
    """
    L = len(dev_probs)
    T = len(dev_probs[0])
    group_ids = assign_groups_by_average_probability(dev_probs, num_groups)
    # Use chosen initialisation strategy for group weights
    group_weights = np.vstack([init_weights(strategy="biased", L=L) for _ in range(num_groups)])
    dev_ppls = []

    best_dev_ppl = float("inf")
    best_group_weights = group_weights.copy()
    best_index = 0

    for i in range(max_iter):
        dev_log_probs = np.zeros(T)

        # E-step: compute interpolated log probs for each group
        for g in range(num_groups):
            mask = (group_ids == g)
            interp_probs = interpolate_probs(dev_probs, group_weights[g])[mask]
            dev_log_probs[mask] = np.log(np.clip(interp_probs, 1e-12, 1.0))

        dev_ppl = compute_perplexity(dev_log_probs)
        dev_ppls.append(dev_ppl)

        # Track best perplexity
        if dev_ppl < best_dev_ppl:
            best_dev_ppl = dev_ppl
            best_group_weights = group_weights.copy()
            best_index = i

        # M-step: update weights for each group
        for g in range(num_groups):
            mask = (group_ids == g)
            posteriors = np.zeros((L, np.sum(mask)))
            for l in range(L):
                posteriors[l] = group_weights[g][l] * (dev_probs[l][mask] ** beta)
            denom = np.sum(posteriors, axis=0)
            posteriors /= denom  # Normalize posteriors

            new_weights = np.sum(posteriors, axis=1) / posteriors.shape[1]

            # Entropy regularisation
            entropy = -np.sum(new_weights * np.log(np.clip(new_weights, 1e-12, 1.0)))
            new_weights += alpha * entropy
            new_weights = np.clip(new_weights, 1e-12, None)
            group_weights[g] = new_weights / np.sum(new_weights)

        # Convergence check (last 5 steps stable)
        if i >= 5 and all(abs(dev_ppls[-j] - dev_ppls[-j-1]) < tol for j in range(1, 5)):
            print(f"Converged at iteration {i}")
            break

    return best_group_weights, dev_ppls, best_index

def plot_grouped_perplexities(dev_ppls, plot_name="grouped_dev_perplexity.png"):
    """
    Saves a plot of perplexity vs. iteration for development data.
    """
    os.makedirs("plots", exist_ok=True)
    plt.plot(dev_ppls, label="Dev Perplexity")
    plt.xlabel("EM Iteration")
    plt.ylabel("Perplexity")
    plt.title("Perplexity Progression (Group-Based EM)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join("plots", plot_name)
    plt.savefig(path)
    plt.close()
    return path

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python estimate.py <dev_glob> <output_weights_file>")
        sys.exit(1)

    dev_glob = sys.argv[1]
    output_path = sys.argv[2]

    dev_files = glob.glob(dev_glob)
    dev_probs = load_ngram_prob_files_additive_with_threshold(dev_files)

    best_group_weights, dev_ppls, best_idx = em_group_weights(dev_probs, alpha=0.0, beta=1.0)

    print("\nGroup-Based EM Weights (best iteration):")
    with open(output_path, "w") as f:
        for g in best_group_weights:
            for w in g:
                f.write(f"{w:.6f}\n")

    # Identify and print top weight
    flat_weights = best_group_weights.flatten()
    top_weight = np.max(flat_weights)
    top_model_index = np.argmax(flat_weights) % len(best_group_weights[0])
    print(f"Top Weight: {top_weight:.3f} (Model {top_model_index + 1})")  # +1 for 1-based indexing

    print("\nWeights saved to:", output_path)
    print("Best perplexity:", dev_ppls[best_idx])
    print("Iterations:", best_idx + 1)
    plot_path = plot_grouped_perplexities(dev_ppls)
    print("Perplexity plot saved to:", plot_path)
