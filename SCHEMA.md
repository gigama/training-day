# data schema

## Structuring Datasets for LLM Training in PyTorch

### Core PyTorch Abstractions

The foundation is PyTorch's `Dataset` and `DataLoader` classes. For LLMs specifically, you'll almost always want a custom `Dataset`:

```python
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.data[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}
```

For large datasets that don't fit in memory, use **iterable datasets** (`IterableDataset`) with streaming, or HuggingFace's `datasets` library which handles memory-mapped files efficiently.

---

### On-Disk Format Matters

Raw text files are inefficient. Better options:

- **Apache Arrow / Parquet** — columnar, memory-mappable, fast random access. HuggingFace datasets uses Arrow natively.
- **WebDataset (tar shards)** — excellent for distributed training; files are sequential and can be streamed from object storage (S3, GCS).
- **LMDB** — key-value store with fast random access, good for single-machine training.
- **Pre-tokenized and saved as `.npy` or `.bin`** — skip tokenization at runtime entirely, which significantly reduces CPU overhead during training.

Pre-tokenizing and packing sequences into fixed-length chunks (e.g., 2048 tokens) with no padding waste is a common production pattern:

```python
# Pack sequences end-to-end with EOS token between documents
# then chunk into fixed-size blocks — no wasted padding tokens
```

---

### Separate Data by Category or Mix It?

This is one of the most consequential decisions in LLM training, and the answer is nuanced.

**The general recommendation is to mix data but control the mixture deliberately through sampling weights**, rather than training sequentially on isolated categories. Here's why, and when exceptions apply:

**Mixing with controlled sampling weights** is preferred because:
- Sequential category training causes **catastrophic forgetting** — the model loses earlier knowledge as it trains on new categories.
- A well-mixed corpus teaches the model to generalize across domains simultaneously.
- You can still express your quality/relevance preferences through **upsampling and downsampling** rather than hard separation.

```python
from torch.utils.data import ConcatDataset, WeightedRandomSampler

high_quality = TextDataset(domain_a_data, tokenizer)
medium_quality = TextDataset(domain_b_data, tokenizer)
low_quality = TextDataset(domain_c_data, tokenizer)

combined = ConcatDataset([high_quality, medium_quality, low_quality])

# Upsample high-quality data 3x, medium 1x, low 0.5x
weights = (
    [3.0] * len(high_quality) +
    [1.0] * len(medium_quality) +
    [0.5] * len(low_quality)
)
sampler = WeightedRandomSampler(weights, num_samples=len(combined))
loader = DataLoader(combined, batch_size=32, sampler=sampler)
```

**When staged or separated training makes sense:**
- **Curriculum learning** — starting with cleaner/simpler data and introducing noisier data later can stabilize early training. This is a deliberate, scheduled mixing strategy, not permanent separation.
- **Domain-adaptive fine-tuning** — after general pretraining, a separate fine-tuning stage on domain-specific data is standard practice (e.g., coding, medicine, law).
- **Instruction tuning / RLHF** — these are almost always separate stages from pretraining, with their own dataset structures.
- **Quality filtering as a gate** — low-quality data shouldn't be in a separate training "phase," it should simply be excluded or heavily downweighted.

---

### Practical Structuring Recommendations

**Metadata tagging over hard separation.** Store data with quality scores, domain labels, and source metadata. This lets you adjust sampling dynamically without restructuring files:

```python
# Each record might look like:
{
  "text": "...",
  "domain": "scientific",
  "quality_score": 0.87,
  "source": "arxiv",
  "token_count": 412
}
```

**Deduplication is critical.** Near-duplicate content in training data causes models to memorize rather than generalize. Tools like MinHash LSH should be run across the full dataset before any splits are made.

**Validation sets should mirror your target distribution.** If the goal is a general-purpose model, validation data should span all domains proportionally. If domain-specific, the validation set should be entirely in-domain and held out strictly.

**Keep a held-out test set untouched** until final evaluation — never use it for hyperparameter tuning or sampling weight decisions.

---

### Summary Heuristic

Mix everything, but be intentional about the mixture ratios. Separate stages are for separate training objectives (pretrain → fine-tune → align), not for segregating content within a single stage. Quality and relevance should be expressed as continuous weights, not binary inclusion/exclusion splits.

## Directory Structure for a Well-Mixed Generalized Training Corpus

The structure below reflects production practices used in large-scale LLM training pipelines. The top-level organization separates **raw ingestion**, **processing stages**, and **training-ready artifacts** — these are distinct concerns and should never be conflated.

```
corpus/
│
├── raw/                          # Immutable source data — never modified after ingestion
│   ├── web/
│   │   ├── cc_2023_06/           # CommonCrawl dump identifier as directory name
│   │   │   ├── shard_0000.warc.gz
│   │   │   ├── shard_0001.warc.gz
│   │   │   └── ...
│   │   └── cc_2024_02/
│   │       ├── shard_0000.warc.gz
│   │       └── ...
│   ├── books/
│   │   ├── gutenberg/
│   │   │   ├── pg1342.txt        # Plain UTF-8 text, filename = PG ID
│   │   │   ├── pg84.txt
│   │   │   └── manifest.jsonl    # {id, title, author, language, license}
│   │   └── open_library/
│   │       ├── ol_001.jsonl
│   │       └── ol_002.jsonl
│   ├── scientific/
│   │   ├── arxiv/
│   │   │   ├── 2301.00001.txt
│   │   │   ├── 2301.00002.txt
│   │   │   └── metadata.parquet  # arXiv metadata: title, abstract, categories, date
│   │   ├── pubmed/
│   │   │   ├── pubmed_baseline_001.xml.gz
│   │   │   └── pubmed_baseline_002.xml.gz
│   │   └── semantic_scholar/
│   │       └── papers_2024.jsonl.gz
│   ├── code/
│   │   ├── github/
│   │   │   ├── python/
│   │   │   │   ├── repo_0000.jsonl  # {repo, path, content, stars, license}
│   │   │   │   └── repo_0001.jsonl
│   │   │   ├── javascript/
│   │   │   │   └── repo_0000.jsonl
│   │   │   ├── rust/
│   │   │   └── go/
│   │   └── stackoverflow/
│   │       └── posts_2024Q1.xml.gz
│   ├── encyclopedic/
│   │   ├── wikipedia/
│   │   │   ├── enwiki_20240101.xml.bz2
│   │   │   └── enwiki_20240101_articles.jsonl
│   │   └── wikidata/
│   │       └── wikidata_20240101.json.gz
│   ├── news/
│   │   ├── cc_news_2023.jsonl.gz
│   │   └── realworld_v2.jsonl
│   ├── multilingual/
│   │   ├── cc_de_2024.jsonl.gz
│   │   ├── cc_fr_2024.jsonl.gz
│   │   ├── cc_zh_2024.jsonl.gz
│   │   └── lang_manifest.csv     # language, source, approx_token_count, script
│   └── dialogue/
│       ├── openhermes_2.5.jsonl
│       └── ultrachat_200k.jsonl
│
├── processed/                    # Derived from raw — reproducible from raw + scripts
│   ├── filtered/                 # After quality filtering, language ID, dedup candidates removed
│   │   ├── web/
│   │   │   ├── shard_0000.jsonl
│   │   │   ├── shard_0001.jsonl
│   │   │   └── filter_stats.json # {input_docs, output_docs, rejection_reasons: {...}}
│   │   ├── books/
│   │   │   └── books_filtered.jsonl
│   │   ├── scientific/
│   │   │   ├── arxiv_filtered.jsonl
│   │   │   └── pubmed_filtered.jsonl
│   │   └── code/
│   │       ├── python_filtered.jsonl
│   │       └── javascript_filtered.jsonl
│   │
│   ├── scored/                   # Quality scores, domain labels, perplexity scores attached
│   │   ├── web_scored.parquet    # Columns: text, domain, quality_score, lang, token_count
│   │   ├── books_scored.parquet
│   │   ├── scientific_scored.parquet
│   │   └── code_scored.parquet
│   │
│   └── deduped/                  # After MinHash LSH or SimHash deduplication
│       ├── web_deduped.parquet
│       ├── books_deduped.parquet
│       ├── scientific_deduped.parquet
│       ├── code_deduped.parquet
│       └── dedup_report.json     # {total_removed, duplicate_pairs_found, method}
│
├── tokenized/                    # Pre-tokenized, packed into fixed-length token blocks
│   ├── pretrain/
│   │   ├── chunk_00000.bin       # Raw token IDs, uint16 or uint32, memory-mappable
│   │   ├── chunk_00001.bin
│   │   ├── chunk_00002.bin
│   │   ├── ...
│   │   └── index.json            # {chunk_count, tokens_per_chunk, total_tokens, vocab}
│   └── validation/
│       ├── val_web.bin
│       ├── val_books.bin
│       ├── val_scientific.bin
│       ├── val_code.bin
│       └── val_index.json
│
├── splits/                       # Train/val/test assignment manifests — no data copied
│   ├── train_manifest.parquet    # Columns: chunk_path, domain, weight, token_count
│   ├── val_manifest.parquet
│   ├── test_manifest.parquet     # Held out — not used until final eval
│   └── sampling_config.yaml     # Mixture weights, upsampling ratios per domain
│
├── metadata/
│   ├── domain_stats.json         # Token counts, document counts per domain
│   ├── quality_thresholds.yaml  # Cutoff scores used during filtering
│   ├── dedup_config.yaml         # MinHash params: num_perm, ngram_size, threshold
│   ├── tokenizer/
│   │   ├── tokenizer.model       # SentencePiece or tiktoken model file
│   │   ├── tokenizer_config.json
│   │   ├── vocab.json
│   │   └── merges.txt            # BPE merge rules if applicable
│   └── licenses/
│       ├── source_licenses.csv   # source, license_type, commercial_use_ok, attribution
│       └── excluded_sources.txt
│
└── pipeline/
    ├── 01_download.py
    ├── 02_extract_text.py
    ├── 03_filter_quality.py
    ├── 04_score_documents.py
    ├── 05_deduplicate.py
    ├── 06_tokenize_and_pack.py
    ├── 07_build_manifests.py
    ├── requirements.txt
    └── README.md
```

---

### Key Design Decisions Reflected Here

**`raw/` is sacred.** Nothing in `raw/` is ever overwritten or deleted. Every downstream artifact is reproducible from `raw/` plus the pipeline scripts. This makes the entire corpus auditable.

**File formats by stage have specific rationale:**
- `.warc.gz` — standard web archive format from CommonCrawl, contains HTTP headers + HTML
- `.jsonl` / `.jsonl.gz` — newline-delimited JSON, streamable without loading full files into memory; one document per line
- `.parquet` — columnar Arrow format; efficient for filtering by `quality_score > 0.7` or `domain == "scientific"` without reading all columns
- `.bin` — flat binary token arrays; memory-mappable with `np.memmap`, zero-copy reads during training
- `.yaml` / `.json` — human-readable config and statistics; never large data

**`splits/` contains manifests, not copies.** Splitting is done by recording which chunks belong to train/val/test in a manifest file. No data is duplicated. The `sampling_config.yaml` looks something like:

```yaml
mixture:
  web:          0.45
  books:        0.15
  scientific:   0.10
  code:         0.15
  encyclopedic: 0.08
  news:         0.04
  multilingual: 0.03

upsample:
  scientific: 2.0    # seen ~2x more often than its raw proportion
  books: 1.5
  code: 1.5
  web: 0.8           # downsampled despite being the largest source
```

**Validation sets are per-domain.** Having separate `val_web.bin`, `val_books.bin`, etc. lets you track loss curves per domain independently during training, which is critical for diagnosing imbalance or forgetting rather than watching a single aggregate loss.

**`metadata/licenses/`** is not optional. Many web and book sources have incompatible commercial licenses. Tracking this at ingestion time prevents legal problems that are very difficult to unwind after training is complete.
