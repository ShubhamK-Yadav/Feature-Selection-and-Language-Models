import sys
import numpy as np
import glob

def load_ngram_prob_files(file_paths, min_prob=1e-9):
    probs = []
    for path in file_paths:
        with open(path) as f:
            values = [max(float(line.strip()), min_prob) for line in f if line.strip()]
            probs.append(np.array(values))
    return probs

def init_weights(strategy, L, top_indices=[0, 5], top_value=0.6):
    if strategy == "uniform":
        return np.full(L, 1.0 / L)
    
    elif strategy == "random":
        weights = np.random.rand(L)
        return weights / np.sum(weights)
    
    elif strategy == "biased":
        weights = np.full(L, (1.0 - top_value) / (L - 1))
        weights[5] = top_value  # favors model 6 (index 5)
        return weights / np.sum(weights)
    
    elif strategy == "top-heavy":
        weights = np.full(L, (1.0 - top_value) / (L - len(top_indices)))
        for i in top_indices:
            weights[i] = top_value / len(top_indices)
        return weights / np.sum(weights)
    
    elif strategy == "bottom-heavy":
        # Assign more weight to models with highest indices
        bottom_indices = list(range(L - len(top_indices), L))
        weights = np.full(L, (1.0 - top_value) / (L - len(bottom_indices)))
        for i in bottom_indices:
            weights[i] = top_value / len(bottom_indices)
        return weights / np.sum(weights)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def interpolate(probs, weights):
    return np.sum([w * p for w, p in zip(weights, probs)], axis=0)

def compute_perplexity(interp_probs):
    return np.exp(-np.mean(np.log(np.clip(interp_probs, 1e-12, 1.0))))

def em(dev_probs, strategy="uniform", max_iter=100, beta=1.0, tol=1e-4, alpha=0.0):
    L = len(dev_probs)
    T = len(dev_probs[0])
    weights = init_weights(strategy, L)
    best_weights = weights.copy()
    best_ppl = float("inf")
    prev_ppl = None
    best_iter = 0

    for i in range(max_iter):
        interp = interpolate(dev_probs, weights)
        log_probs = np.log(np.clip(interp, 1e-12, 1.0))
        ppl = compute_perplexity(np.exp(log_probs))

        if ppl < best_ppl:
            best_ppl = ppl
            best_weights = weights.copy()
            best_iter = i + 1  # human-readable iteration count

        # E-step
        posteriors = np.array([w * (p ** beta) for w, p in zip(weights, dev_probs)])
        posteriors /= np.sum(posteriors, axis=0)

        # M-step
        new_weights = np.sum(posteriors, axis=1) / T
        entropy = -np.sum(new_weights * np.log(np.clip(new_weights, 1e-12, 1.0)))
        new_weights += alpha * entropy
        new_weights = np.clip(new_weights, 1e-12, None)
        weights = new_weights / np.sum(new_weights)

        if prev_ppl is not None and abs(prev_ppl - ppl) < tol:
            break
        prev_ppl = ppl

    return best_weights, best_ppl, best_iter


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python estimate.py <dev_glob> <init_strategy> <output_weights_file>")
        sys.exit(1)

    dev_glob = sys.argv[1]
    strategy = sys.argv[2]
    output_path = sys.argv[3]

    dev_files = glob.glob(dev_glob)
    dev_probs = load_ngram_prob_files(dev_files)

    weights, dev_ppl, iterations = em(dev_probs, strategy=strategy, max_iter=100)

    top_index = np.argmax(weights)
    top_value = weights[top_index]

    print(f"\nStrategy: {strategy}")
    print(f"Iterations: {iterations}")
    print(f"Top weight: {top_value:.4f} (Model {top_index + 1})")
    for i, w in enumerate(weights):
        print(f"λ[{i}] = {w:.6f}")
    print(f"Final dev perplexity: {dev_ppl:.4f}")

