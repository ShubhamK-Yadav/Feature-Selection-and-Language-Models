# Keyword Search (KWS) System for Spoken Data
## Overview
This project implements a keyword search (KWS) system for spoken data, designed to index, search, and evaluate keyword occurrences from CTM files. It uses gap-tolerant matching strategies and applies evaluation metrics to assess retrieval performance.

### System Components
#### 1. Indexing (index.py)
Parses CTM files into structured dictionaries for efficient keyword lookup.

Extracts:
- File ID
- Start time and end time (computed from start time + duration)
- Lowercased wordConfidence score (defaults to 1.0 if missing)
- Stores processed entries grouped by file in a pickled dictionary for fast retrieval.

#### 2. Search & Retrieval (search.py)
Reads the keyword list and the indexed data. Implements gap-tolerant matching for multi-word keywords:

**Single-word:** exact match.
**Multi-word:** allows insertions and mismatches between tokens.
Score matches using the mean token confidence.

Applies **Sum-To-One (STO) normalisation** with γ = 1.0:

```
r̂_i,j = r_i,j^γ / Σ_k r_i,k^γ
```

**Thresholding:**
- Per-keyword 20th percentile threshold.
- Global minimum confidence threshold: 0.05.

#### 3. Matching Strategy
##### Preprocessing:
- Tokenises and lowercases keywords.
- Builds a word-to-position map for each file.

##### Matching:
**Single-word**: exact match.
**Multi-word**: gap-tolerant, allowing flexibility for ASR errors.
**Scoring**:
Confidence score = mean of token-level confidences.

#### 4. Evaluation
##### Matching Criterion
A keyword detection is correct if start and end times fall within ±0.5 seconds of a reference occurrence in the STM file.
This accounts for small timing variations caused by ASR segmentation.

#### Metrics
The system is evaluated using:

**AUC (Area Under Curve):**
Quality of ranking via the precision–recall curve.

**mAP (Mean Average Precision):**
Average ranking consistency across all queries.

**TWV (Term-Weighted Value):
**Operational effectiveness, penalising missed detections and false alarms:

```
TWV = 1 − (Pmiss + βPfa), β = 20
```

### Usage
#### Indexing
```
python index.py --input path/to/ctm/files --output index.pkl
```

#### Searching
```
python search.py --keywords keywords.txt --index index.pkl --output results.txt
```

#### Evaluation
```
python evaluate.py --refs references.stm --results results.txt
```
