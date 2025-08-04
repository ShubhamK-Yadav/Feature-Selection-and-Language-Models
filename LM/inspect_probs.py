import sys

file_path = sys.argv[1]

with open(file_path, 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"File: {file_path}")
print(f"Total entries: {len(lines)}")
print("First 10 values:")
for line in lines[:10]:
    print(line)
