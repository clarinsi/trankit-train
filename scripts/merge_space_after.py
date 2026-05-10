#!/usr/bin/env python3
"""Transfer SpaceAfter=No from an original CoNLL-U into a Trankit-annotated CoNLL-U.

Trankit sets MISC=_ for all tokens. This script reads the original tokenised
CoNLL-U (which has SpaceAfter=No where applicable) and copies that information
into the annotated file, sentence by sentence, token by token.

Usage:
    python scripts/merge_space_after.py \
        --original  data/datasets/ccKres/cckres.conllu \
        --annotated data/datasets/ccKres/cckres_annotated.conllu \
        --output    data/datasets/ccKres/cckres_annotated.conllu
"""

import argparse
import logging
import os
import sys
import tempfile

import conllu


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stderr)])
    return logging.getLogger(__name__)


def merge(original_path, annotated_path, output_path, logger):
    sent_count = 0
    mismatch_sents = 0
    merged_tokens = 0

    # Write to a temp file first to safely handle in-place operation
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(output_path)))
    try:
      with open(original_path, encoding='utf-8') as orig_f, \
           open(annotated_path, encoding='utf-8') as ann_f, \
           open(tmp_fd, 'w', encoding='utf-8') as out:

        orig_iter = conllu.parse_incr(orig_f)
        ann_iter  = conllu.parse_incr(ann_f)

        for orig_tl, ann_tl in zip(orig_iter, ann_iter):
            if len(orig_tl) != len(ann_tl):
                logger.warning(
                    f'sent_id={ann_tl.metadata.get("sent_id","?")}: '
                    f'token count mismatch orig={len(orig_tl)} ann={len(ann_tl)}, skipping SpaceAfter'
                )
                mismatch_sents += 1
                out.write(ann_tl.serialize())
                sent_count += 1
                continue

            for orig_tok, ann_tok in zip(orig_tl, ann_tl):
                orig_misc = orig_tok.get('misc') or {}
                if orig_misc.get('SpaceAfter') == 'No':
                    if ann_tok['misc'] is None:
                        ann_tok['misc'] = {}
                    ann_tok['misc']['SpaceAfter'] = 'No'
                    merged_tokens += 1

            out.write(ann_tl.serialize())
            sent_count += 1

            if sent_count % 50000 == 0:
                logger.info(f'{sent_count} sentences processed')

      os.replace(tmp_path, output_path)
    except Exception:
      os.unlink(tmp_path)
      raise

    logger.info(
        f'Done. {sent_count} sentences, {merged_tokens} SpaceAfter=No tokens merged, '
        f'{mismatch_sents} length-mismatch sentences → {output_path}'
    )


def main():
    parser = argparse.ArgumentParser(description='Merge SpaceAfter into annotated CoNLL-U')
    parser.add_argument('--original',  required=True, help='Original tokenised CoNLL-U (has SpaceAfter)')
    parser.add_argument('--annotated', required=True, help='Trankit-annotated CoNLL-U (MISC=_)')
    parser.add_argument('--output',    required=True, help='Output CoNLL-U')
    args = parser.parse_args()

    logger = setup_logging()
    merge(args.original, args.annotated, args.output, logger)


if __name__ == '__main__':
    main()
