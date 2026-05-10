#!/usr/bin/env python3
"""Annotate ccKres CoNLL-U with Trankit (step 2).

Reads pre-tokenised CoNLL-U (from cckres_tei_to_conllu.py), runs Trankit
in chunks, and writes predictions incrementally with progress logging.

Usage:
    python scripts/cckres_annotate.py \
        --input    data/datasets/ccKres/cckres.conllu \
        --output   data/datasets/ccKres/cckres_annotated.conllu \
        --save_dir data/models/save_dir_ssj2.14+sst2.15-stan+pog \
        --embedding xlm-roberta-large \
        --chunk_size 1000
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import conllu

# Fix dead HuggingFace CDN URLs bundled with trankit's adapter_transformers
def _fixed_hf_bucket_url(identifier, filename, use_cdn=False, mirror=None):
    return f"https://huggingface.co/{identifier}/resolve/main/{filename}"

import trankit.adapter_transformers.modeling_utils as _mu
import trankit.adapter_transformers.configuration_utils as _cu
import trankit.adapter_transformers.tokenization_utils as _tu
_mu.hf_bucket_url = _fixed_hf_bucket_url
_cu.hf_bucket_url = _fixed_hf_bucket_url
_tu.hf_bucket_url = _fixed_hf_bucket_url

from trankit import Pipeline, trankit2conllu


def setup_logging(log_file=None):
    handlers = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=handlers,
    )
    return logging.getLogger(__name__)


def load_sentences(conllu_path):
    """Yield parsed TokenList objects from a CoNLL-U file."""
    with open(conllu_path, encoding='utf-8') as f:
        for tl in conllu.parse_incr(f):
            yield tl


def annotate_chunk(pipeline, sentences):
    """Run Trankit on a list of TokenList objects, return annotated CoNLL-U string."""
    token_lists = [[tok['form'] for tok in tl] for tl in sentences]
    result = pipeline(token_lists)
    return trankit2conllu(result)


def merge_metadata(annotated_conllu_str, original_sentences):
    """Re-attach original metadata (sent_id, text, newdoc, newpar) to annotated output."""
    annotated = conllu.parse(annotated_conllu_str)
    out_lines = []

    for orig, pred in zip(original_sentences, annotated):
        # Write original metadata comments in order
        for key, val in orig.metadata.items():
            out_lines.append(f'# {key} = {val}')
        # Write predicted token lines (skip auto-generated comments from trankit2conllu)
        pred_lines = pred.serialize().splitlines()
        for line in pred_lines:
            if not line.startswith('#'):
                out_lines.append(line)

    return '\n'.join(out_lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description='Annotate ccKres CoNLL-U with Trankit')
    parser.add_argument('--input',     default='data/datasets/ccKres/cckres.conllu')
    parser.add_argument('--output',    default='data/datasets/ccKres/cckres_annotated.conllu')
    parser.add_argument('--save_dir',  default='data/models/save_dir_ssj2.14+sst2.15-stan+pog')
    parser.add_argument('--embedding', default='xlm-roberta-large')
    parser.add_argument('--category',  default='customized')
    parser.add_argument('--chunk_size', type=int, default=1000,
                        help='Sentences per Trankit batch')
    parser.add_argument('--log', default=None)
    args = parser.parse_args()

    logger = setup_logging(args.log)

    logger.info(f'Loading pipeline: {args.save_dir} ({args.embedding})')
    pipeline = Pipeline(lang=args.category, cache_dir=args.save_dir, embedding=args.embedding)

    logger.info(f'Reading {args.input}')
    total_sents = 0
    total_toks = 0
    chunk = []
    t_start = time.time()

    with open(args.output, 'w', encoding='utf-8') as out:
        for tl in load_sentences(args.input):
            chunk.append(tl)

            if len(chunk) >= args.chunk_size:
                annotated = annotate_chunk(pipeline, chunk)
                merged = merge_metadata(annotated, chunk)
                out.write(merged)

                total_sents += len(chunk)
                total_toks  += sum(len(tl) for tl in chunk)
                elapsed = time.time() - t_start
                rate = total_sents / elapsed
                logger.info(
                    f'{total_sents} sentences, {total_toks} tokens '
                    f'({rate:.0f} sents/s, {elapsed:.0f}s elapsed)'
                )
                chunk = []

        # Final partial chunk
        if chunk:
            annotated = annotate_chunk(pipeline, chunk)
            merged = merge_metadata(annotated, chunk)
            out.write(merged)
            total_sents += len(chunk)
            total_toks  += sum(len(tl) for tl in chunk)

    elapsed = time.time() - t_start
    logger.info(
        f'Done. {total_sents} sentences, {total_toks} tokens in {elapsed:.0f}s '
        f'→ {args.output}'
    )


if __name__ == '__main__':
    main()
