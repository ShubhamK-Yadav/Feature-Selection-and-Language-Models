import sys
import re
import pickle
import csv
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor


def load_keywords(keyword_path):
    with open(keyword_path, 'r') as f:
        return [line.strip().lower() for line in f if line.strip()]

def preprocess_keywords(keywords):
    return [(kw, kw.split()) for kw in keywords]

def load_index(index_path):
    with open(index_path, 'rb') as f:
        return pickle.load(f)

def matches_within_gaps(word_seq, token_seq, max_gap=1):
    i = 0
    for token in token_seq:
        found = False
        for j in range(i, len(word_seq)):
            if word_seq[j] == token:
                found = True
                i = j + 1
                break
            if j - i > max_gap:
                return False
        if not found:
            return False
    return True

def build_word_positions(entries):
    word_positions = defaultdict(list)
    for i, (_, _, word, _) in enumerate(entries):
        word_positions[word].append(i)
    return word_positions

def search_single_keyword(file_id, entries, preprocessed_keywords, max_gap=1):
    results = []
    entries.sort()
    words = [e[2] for e in entries]
    word_positions = build_word_positions(entries)
    for kw, tokens in preprocessed_keywords:
        if len(tokens) == 1:
            for start, end, word, conf in entries:
                if word == kw:
                    results.append((kw, file_id, start, end, conf))
        else:
            first_token = tokens[0]
            for i in word_positions.get(first_token, []):
                window = words[i:i + len(tokens) + max_gap]
                if matches_within_gaps(window, tokens, max_gap):
                    seg = entries[i:i + len(tokens)]
                    start = seg[0][0]
                    end = seg[-1][1]
                    conf = np.mean([e[3] for e in seg])
                    results.append((kw, file_id, start, end, conf))
    return results

def search_keywords(index, preprocessed_keywords, max_gap=1):
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(search_single_keyword, file_id, entries, preprocessed_keywords, max_gap)
                   for file_id, entries in index.items()]
        for f in futures:
            results.extend(f.result())
    return results

def sto_normalize(results, gamma=2.0):
    keyword_scores = defaultdict(list)
    for r in results:
        keyword_scores[r[0]].append(r)
    normalized = []
    for kw, occs in keyword_scores.items():
        denom = sum((r[4] ** gamma) for r in occs)
        for r in occs:
            sto_score = (r[4] ** gamma) / denom if denom > 0 else 0.0
            normalized.append(r[:4] + (sto_score,))
    return normalized

def compute_keyword_specific_thresholds(results, percentile=40):
    keyword_to_scores = defaultdict(list)
    for r in results:
        keyword_to_scores[r[0]].append(r[4])
    thresholds = {kw: np.percentile(scores, percentile) for kw, scores in keyword_to_scores.items()}
    return thresholds

def apply_keyword_thresholds(results, thresholds, abs_min=0.15):
    return [r for r in results if r[4] >= max(thresholds.get(r[0], 0), abs_min)]

def save_occurrences(results, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["keyword", "file", "start", "end", "confidence"])
        for row in results:
            writer.writerow(row)

def main():
    if len(sys.argv) != 4:
        print("Usage: python search.py <index_file> <keyword_file> <output_file>")
        return

    index_file = sys.argv[1]
    keyword_file = sys.argv[2]
    output_file = sys.argv[3]

    index = load_index(index_file)
    keywords = load_keywords(keyword_file)
    preprocessed_keywords = preprocess_keywords(keywords)

    results = search_keywords(index, preprocessed_keywords, max_gap=0)
    results = sto_normalize(results, gamma=0.5)
    thresholds = compute_keyword_specific_thresholds(results, percentile=10)
    results = apply_keyword_thresholds(results, thresholds, abs_min=0.05)
    save_occurrences(results, output_file)

    print(f"[INFO] Optimized search complete. Results written to {output_file} ({len(results)} entries)")

if __name__ == "__main__":
    main()
