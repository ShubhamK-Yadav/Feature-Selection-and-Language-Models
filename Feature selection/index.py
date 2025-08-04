# index.py
import sys
import re
import pickle
from collections import defaultdict

def load_ctm_file(ctm_path):
    print(f"[INFO] Loading CTM file from {ctm_path} ...")
    index = defaultdict(list)
    with open(ctm_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            file_id = parts[0]
            start_time = float(parts[2])
            duration = float(parts[3])
            end_time = start_time + duration
            word = parts[4].lower()
            confidence = float(parts[5]) if len(parts) > 5 else 1.0
            index[file_id].append((start_time, end_time, word, confidence))
    print(f"[INFO] Loaded {len(index)} files from CTM.")
    return index

def save_index(index, output_path):
    print(f"[INFO] Saving index to {output_path} ...")
    with open(output_path, 'wb') as f:
        pickle.dump(index, f)
    print("[INFO] Index saved successfully.")

def main():
    if len(sys.argv) != 3:
        print("Usage: python index.py <ctm_file> <index_file>")
        return
    ctm_path = sys.argv[1]
    index_path = sys.argv[2]

    index = load_ctm_file(ctm_path)
    save_index(index, index_path)
    print(f"Index saved to {index_path}")


if __name__ == "__main__":
    main()