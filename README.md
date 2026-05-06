# pytorch-train

Pipeline for generating PyTorch training datasets from raw text files.

## Overview

**`training-day.py`** — Tokenise bundles into 1,000-record PyTorch JSON datasets with optimized word relationships.

---

## Setup

```bash
bash setup.sh
source pytorch-env/bin/activate
```

Requires Python 3.12+. Creates a `pytorch-env/` venv and installs dependencies from `requirements.txt`.

---

## Usage

### Step 2a — Tokenise (`training-day.py`)

```bash
python3 training-day.py --input text-out/ --output training-data/ --type True
python3 training-day.py --input text-out/ --output training-data/ --type False
```

- Preserves natural sentence context from the ingested input
- `--type` tags every record in the output dataset: `True` → label=1, `False` → label=0
- Output 1,000-record PyTorch JSON files; incomplete bundles are discarded

---

## Output Format

Each output JSON file contains 1,000 total records (800 train / 200 test) and is ready to import directly into PyTorch:

```json
{
  "dataset_info": {
    "total_samples": 1000,
    "train_samples": 800,
    "test_samples": 200,
    "vocab_size": 3090,
    "max_sequence_length": 12,
    "label_distribution": {
      "train": {"0": 0, "1": 800},
      "test":  {"0": 0, "1": 200},
      "total": {"0": 0, "1": 1000}
    },
    "creation_timestamp": "2026-04-01T01:01:00",
    "random_seed": 42
  },
  "vocabulary": {
    "token_to_index": {"<PAD>": 0, "<UNK>": 1, "word": 2},
    "index_to_token": {"0": "<PAD>", "1": "<UNK>", "2": "word"},
    "special_tokens": {"pad_token": "<PAD>", "unk_token": "<UNK>", "pad_id": 0, "unk_id": 1}
  },
  "train_data": [{"id": 0, "text": "...", "tokens": [2, 5, 11], "label": 1}],
  "test_data":  [{"id": 0, "text": "...", "tokens": [3, 7, 2],  "label": 1}],
  "metadata": {
    "source_files": ["..."],
    "processing_parameters": {"method": "tokenize", "type": "True", "bundle_size": 1000}
  }
}
```

---

## Dependencies

| Package     | Version      |
|-------------|--------------|
| torch       | 2.11.0+cpu   |
| torchvision | 0.26.0+cpu   |
| pandas      | 3.0.2        |
| numpy       | 2.4.4        |
