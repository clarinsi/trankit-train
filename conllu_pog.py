'''
Replaces standardized word forms (FORM column) with spoken/colloquial pronunciation
variants from the MISC field (pronunciation=X). Used to create pronunciation-based
variants of CoNLLu datasets for training Trankit models on colloquial Slovenian speech.

Optionally produces a shuffled combined file (standardized + pronunciation sentences)
via --combined_output_fpath, which is recommended for fine-tuning to avoid recency bias.

Usage:
    # Create pronunciation variant only
    python conllu_pog.py \
        --conllu_input_fpath data/ssj2.14+sst2.15-dev3/sl_ssj+sst-ud-train-formatted.conllu \
        --conllu_output_fpath data/ssj2.14+sst2.15-dev3/sl_ssj+sst-ud-train-pog.conllu

    # Also create shuffled combined file (stan + pog)
    python conllu_pog.py \
        --conllu_input_fpath data/ssj2.14+sst2.15-dev3/sl_ssj+sst-ud-train-formatted.conllu \
        --conllu_output_fpath data/ssj2.14+sst2.15-dev3/sl_ssj+sst-ud-train-pog.conllu \
        --combined_output_fpath data/ssj2.14+sst2.15-dev3/sl_ssj+sst-ud-train-stan+pog.conllu
'''

import argparse
import copy
import logging
import os
import random
import time

from conllu import parse_incr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def read_args():
    parser = argparse.ArgumentParser(
        description='Replace FORM with pronunciation variant from MISC field.'
    )
    parser.add_argument('--conllu_input_fpath', type=str, required=True,
                        help='Input CoNLLu file (standardized forms)')
    parser.add_argument('--conllu_output_fpath', type=str, required=True,
                        help='Output CoNLLu file with pronunciation forms (-pog)')
    parser.add_argument('--combined_output_fpath', type=str, default=None,
                        help='Optional: output shuffled combined file (standardized + pronunciation)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling combined output (default: 42)')
    return parser.parse_args()


def get_pronunciation(misc):
    '''Extract pronunciation value from token MISC field (dict or None).'''
    if misc is None:
        return None
    # conllu library parses MISC as an OrderedDict for key=value pairs
    if isinstance(misc, dict):
        return misc.get('pronunciation')
    return None


def has_space_after(misc):
    '''Return False if SpaceAfter=No is set in MISC.'''
    if misc is None:
        return True
    if isinstance(misc, dict):
        return misc.get('SpaceAfter') != 'No'
    return True


def is_regular_token(token_id):
    '''Return True for regular tokens; False for multi-word (1-2) and empty nodes (1.1).'''
    return isinstance(token_id, int)


def apply_pronunciation(tokenlist):
    '''
    Return a deep copy of tokenlist with FORM replaced by pronunciation variant
    and # text = metadata reconstructed from the new forms.
    For tokens without a pronunciation annotation (e.g. SSJ portion), FORM is kept as-is.
    '''
    tl = copy.deepcopy(tokenlist)
    text_parts = []

    for token in tl:
        if not is_regular_token(token['id']):
            continue
        pron = get_pronunciation(token['misc'])
        if pron is not None:
            token['form'] = pron
        text_parts.append((token['form'], has_space_after(token['misc'])))

    # Reconstruct sentence text from (possibly updated) forms
    text = ''
    for i, (form, space_after) in enumerate(text_parts):
        text += form
        if space_after and i < len(text_parts) - 1:
            text += ' '
    tl.metadata['text'] = text

    return tl


def write_sentences(sentences, fpath):
    if os.path.exists(fpath):
        os.remove(fpath)
    with open(fpath, 'w', encoding='utf-8') as f:
        for tl in sentences:
            f.write(tl.serialize())


def main():
    args = read_args()

    logger.info('Reading: %s', args.conllu_input_fpath)
    stan_sentences = []
    with open(args.conllu_input_fpath, 'r', encoding='utf-8') as f:
        for tl in parse_incr(f):
            stan_sentences.append(tl)
    logger.info('Loaded %d sentences', len(stan_sentences))

    pog_sentences = [apply_pronunciation(tl) for tl in stan_sentences]

    logger.info('Writing pronunciation output: %s', args.conllu_output_fpath)
    write_sentences(pog_sentences, args.conllu_output_fpath)
    logger.info('Wrote %d sentences', len(pog_sentences))

    if args.combined_output_fpath:
        combined = stan_sentences + pog_sentences
        random.seed(args.seed)
        random.shuffle(combined)
        logger.info(
            'Writing shuffled combined output (%d sentences, seed=%d): %s',
            len(combined), args.seed, args.combined_output_fpath
        )
        write_sentences(combined, args.combined_output_fpath)
        logger.info('Done.')


if __name__ == '__main__':
    start_time = time.time()
    main()
    logger.info('Total: %.0fs', time.time() - start_time)
