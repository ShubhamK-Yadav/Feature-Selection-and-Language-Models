# estimate.py: EM with entropy regularisation using only dev data
import sys
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def load_ngram_prob_files_additive_with_threshold(file_paths, min_prob=1e-9):
    all_probs = []
    for path in file_paths:
        with open(path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            values = [max(float(line), min_prob) for line in lines]
            all_probs.append(np.array(values))
    return all_probs

def interpolate_probs(probs, weights):
    return np.sum([w * p for w, p in zip(weights, probs)], axis=0)

def compute_perplexity(log_probs):
    return np.exp(-np.mean(log_probs))

def em_with_entropy_regularisation(dev_probs, beta=1.0, max_iter=100, tol=1e-4, alpha=0.0001, init_weights=None):
    L = len(dev_probs)
    weights = np.full(L, 1.0 / L) if init_weights is None else np.array(init_weights)
    dev_ppls, lambda_history = [], []

    best_dev_ppl = float("inf")
    best_lambda = weights.copy()
    best_index = 0

    for i in range(max_iter):
        lambda_history.append(weights.copy())
        interpolated_dev = interpolate_probs(dev_probs, weights)
        log_dev = np.log(np.clip(interpolated_dev, 1e-12, 1.0))
        dev_ppl = compute_perplexity(log_dev)
        dev_ppls.append(dev_ppl)

        if dev_ppl < best_dev_ppl:
            best_dev_ppl = dev_ppl
            best_lambda = weights.copy()
            best_index = i

        posteriors = np.zeros((L, len(dev_probs[0])))
        for l in range(L):
            posteriors[l] = weights[l] * (dev_probs[l] ** beta)
        posteriors /= np.sum(posteriors, axis=0)

        new_weights = np.sum(posteriors, axis=1) / posteriors.shape[1]
        entropy = -np.sum(new_weights * np.log(np.clip(new_weights, 1e-12, 1.0)))
        new_weights += alpha * entropy
        new_weights = np.clip(new_weights, 1e-12, None)
        weights = new_weights / np.sum(new_weights)

        if i > 0 and abs(dev_ppls[-1] - dev_ppls[-2]) < tol:
            break

    return best_lambda, dev_ppls, lambda_history, best_index

# def plot_dev_perplexity(dev_ppls):
#     os.makedirs("plots", exist_ok=True)
#     plt.plot(dev_ppls, label="Dev Perplexity")
#     plt.xlabel("EM Iteration")
#     plt.ylabel("Perplexity")
#     plt.title("Perplexity Progression (Dev)")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     path = "plots/dev_perplexity_progression.png"
#     plt.savefig(path)
#     plt.close()
#     return path

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python estimate.py <dev_glob> <output_weights_file>")
        sys.exit(1)

    dev_glob = sys.argv[1]
    output_path = sys.argv[2]

    dev_files = glob.glob(dev_glob)
    dev_probs = load_ngram_prob_files_additive_with_threshold(dev_files)

    best_weights, dev_ppls, _, best_idx = em_with_entropy_regularisation(dev_probs)
    # plot_path = plot_dev_perplexity(dev_ppls)

    print("\n✅ Best weights (to be saved):")
    for i, w in enumerate(best_weights):
        print(f"λ[{i}] = {w:.6f}")

    with open(output_path, "w") as f:
        for w in best_weights:
            f.write(f"{w:.6f}\n")

    print("\n✅ Best weights saved to:", output_path)
    print("Best dev perplexity:", dev_ppls[best_idx])
    print("Iterations:", best_idx + 1)
    # print("Perplexity plot saved to:", plot_path)
