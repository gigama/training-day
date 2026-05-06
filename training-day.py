#!/usr/bin/env python3
"""
training-day.py - Tokenize text files into PyTorch training datasets.

Rearranges tokens to maximise word co-occurrence relationships.
Command line parameters:
    '--type' tags the dataset as True (label=1) or False (label=0)
    '--input specifies input directory of text files
    '--output specifies output directory for PyTorch JSON files'
Outputs 1,000-record PyTorch JSON files; incomplete bundles are discarded.
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TextTokenDataset(Dataset):
    def __init__(self, texts, labels, vocab=None):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.vocab = vocab if vocab is not None else self._build_vocab(texts)
        self.tokenized = [self._tokenize(t) for t in texts]

    def _build_vocab(self, texts):
        vocab = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        for text in texts:
            for token in text.lower().split():
                if token not in vocab:
                    vocab[token] = idx
                    idx += 1
        return vocab

    def _tokenize(self, text):
        return [self.vocab.get(t, self.vocab['<UNK>']) for t in text.lower().split()]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'tokens': torch.tensor(self.tokenized[idx], dtype=torch.long),
            'label': self.labels[idx]
        }


_CONSEC_DUP = re.compile(r'\b(\w+)\s+\1\b', re.IGNORECASE)

def post_process(sentences):
    """Remove duplicate sentences and sentences with consecutively repeated words."""
    seen = set()
    result = []
    for sentence in sentences:
        if sentence in seen:
            continue
        if _CONSEC_DUP.search(sentence):
            continue
        seen.add(sentence)
        result.append(sentence)
    return result


def extract_tokens(text):
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())


def calculate_cooccurrence(tokens, window_size=5):
    matrix = defaultdict(lambda: defaultdict(int))
    for i, token in enumerate(tokens):
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        for j in range(start, end):
            if i != j:
                matrix[token][tokens[j]] += 1
    return matrix


def rel_score(w1, w2, matrix):
    return matrix[w1][w2] + matrix[w2][w1]


def create_related_sentences(tokens, matrix, max_words=10):
    """Build sentences by selecting tokens with co-occurrence scores."""
    sentences = []
    remaining = tokens[:]
    random.shuffle(remaining)

    while remaining:
        sent = []
        available = remaining[:]

        if available:
            first = available.pop(0)
            sent.append(first)
            remaining.remove(first)

        while len(sent) < max_words and available:
            best, lowest = None, float('inf')
            for candidate in available:
                score = sum(rel_score(candidate, w, matrix) for w in sent)
                if score < lowest:
                    lowest, best = score, candidate
            if best:
                sent.append(best)
                available.remove(best)
                remaining.remove(best)
            else:
                break

        if sent:
            sentences.append(' '.join(sent))

    return sentences


def build_pytorch_json(sentences, labels, source_files, label_tag, random_seed=42):
    """Build the full PyTorch-ready JSON structure from exactly 1,000 records."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    combined = list(zip(sentences, labels))
    random.shuffle(combined)
    sentences, labels = map(list, zip(*combined))

    n_total = len(sentences)
    n_test = max(1, int(n_total * 0.2))
    n_train = n_total - n_test

    train_sents = sentences[:n_train]
    train_labels = labels[:n_train]
    test_sents = sentences[n_train:]
    test_labels = labels[n_train:]

    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for text in train_sents:
        for token in text.lower().split():
            if token not in vocab:
                vocab[token] = idx
                idx += 1

    def tokenize(text):
        return [vocab.get(t, vocab['<UNK>']) for t in text.lower().split()]

    train_data = [
        {'id': i, 'text': s, 'tokens': tokenize(s), 'label': l}
        for i, (s, l) in enumerate(zip(train_sents, train_labels))
    ]
    test_data = [
        {'id': i, 'text': s, 'tokens': tokenize(s), 'label': l}
        for i, (s, l) in enumerate(zip(test_sents, test_labels))
    ]

    all_records = train_data + test_data
    max_seq_len = max((len(r['tokens']) for r in all_records), default=0)

    train_0 = train_labels.count(0)
    train_1 = train_labels.count(1)
    test_0 = test_labels.count(0)
    test_1 = test_labels.count(1)

    return {
        'dataset_info': {
            'total_samples': n_total,
            'train_samples': n_train,
            'test_samples': n_test,
            'vocab_size': len(vocab),
            'max_sequence_length': max_seq_len,
            'label_distribution': {
                'train': {'0': train_0, '1': train_1},
                'test': {'0': test_0, '1': test_1},
                'total': {'0': train_0 + test_0, '1': train_1 + test_1}
            },
            'creation_timestamp': datetime.now().isoformat(),
            'random_seed': random_seed
        },
        'vocabulary': {
            'token_to_index': vocab,
            'index_to_token': {str(v): k for k, v in vocab.items()},
            'special_tokens': {
                'pad_token': '<PAD>',
                'unk_token': '<UNK>',
                'pad_id': 0,
                'unk_id': 1
            }
        },
        'train_data': train_data,
        'test_data': test_data,
        'metadata': {
            'source_files': source_files,
            'processing_parameters': {
                'method': 'tokenize',
                'type': label_tag,
                'bundle_size': 1000,
                'test_split': 0.2,
                'random_seed': random_seed
            },
            'original_text_stats': {
                'total_sentences': n_total
            }
        }
    }


def read_input_files(input_dir):
    """Read all .md and .txt files; return list of non-empty lines."""
    sentences = []
    files = sorted(list(input_dir.glob('*.md')) + list(input_dir.glob('*.txt')))
    if not files:
        print(f"No .md or .txt files found in {input_dir}")
        sys.exit(1)
    for fp in files:
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line:
                    sentences.append(line)
    return sentences, [fp.name for fp in files]


def main():
    parser = argparse.ArgumentParser(
        description='Tokenize ingested text files into PyTorch training datasets'
    )
    parser.add_argument('--input', required=True,
                        help='Input directory of text files')
    parser.add_argument('--output', required=True,
                        help='Output directory for PyTorch JSON files')
    parser.add_argument('--type', required=True, choices=['True', 'False'],
                        help='Dataset label tag: True (label=1) or False (label=0)')
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    label_value = 1 if args.type == 'True' else 0

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading input files from: {input_dir}")
    sentences, source_files = read_input_files(input_dir)
    print(f"Total lines loaded: {len(sentences)}")

    random.shuffle(sentences)

    print(f"Tokenizing (label={label_value})...")
    all_text = ' '.join(sentences)
    tokens = extract_tokens(all_text)

    if not tokens:
        print("No tokens found in input. No output written.")
        sys.exit(1)

    print(f"  Building co-occurrence matrix ({len(tokens)} tokens)...")
    matrix = calculate_cooccurrence(tokens)

    print(f"  Generating related sentences...")
    raw = create_related_sentences(tokens, matrix)
    raw = [s[0].upper() + s[1:] if s else s for s in raw]

    processed = post_process(raw)
    labels = [label_value] * len(processed)
    print(f"Records after post-processing: {len(processed)}")

    bundle_size = 1000
    if len(processed) < bundle_size:
        print(f"Insufficient records ({len(processed)} < {bundle_size}). No output written.")
        sys.exit(0)

    written = 0
    for i in range(0, len(processed), bundle_size):
        chunk_s = processed[i:i + bundle_size]
        chunk_l = labels[i:i + bundle_size]

        if len(chunk_s) < bundle_size:
            print(f"Skipping incomplete bundle ({len(chunk_s)} records).")
            continue

        data = build_pytorch_json(chunk_s, chunk_l, source_files, args.type)
        out_file = output_dir / f"pytorch-{written + 1:04d}.json"

        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        info = data['dataset_info']
        print(f"  Written: {out_file.name} "
              f"(train={info['train_samples']}, test={info['test_samples']}, "
              f"vocab={info['vocab_size']})")
        written += 1

    if written == 0:
        print("No complete 1,000-record bundles could be written.")
    else:
        print(f"\nCreated {written} PyTorch dataset file(s) in {output_dir}")


if __name__ == '__main__':
    main()
