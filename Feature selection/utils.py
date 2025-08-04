import csv
from collections import defaultdict

def is_match(pred_start, pred_end, ref_start, ref_end, tol=0.5):
    return abs(pred_start - ref_start) <= tol and abs(pred_end - ref_end) <= tol

def load_occurrences(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        return [(r["keyword"], r["file"], float(r["start"]), float(r["end"]), float(r["confidence"])) for r in reader]

def load_refs_from_stm(path, keywords=None):
    refs = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            file_id = parts[0]
            start = float(parts[3])
            end = float(parts[4])
            words = parts[6:]
            if not words:
                continue
            duration = end - start
            word_duration = duration / len(words)
            for i, word in enumerate(words):
                word = word.lower()
                if keywords and word not in keywords:
                    continue
                word_start = start + i * word_duration
                word_end = word_start + word_duration
                refs.append((word, file_id, word_start, word_end))
    return refs

def group_by_length(items):
    grouped = defaultdict(list)
    for k, *_ in items:
        length = len(k.split())
        grouped[length].append(k)
    return grouped