#!/usr/bin/env python3
"""Convert GOS TEI XML files to CoNLL-U for Trankit annotation.

Reads Gos.TEI/**/*.xml, outputs a single CoNLL-U file preserving original
sentence and word segmentation.

Token inclusion:
  - <w>  regular words (including those inside <del> false starts)
  - <pc> punctuation
  - <gap>, <vocal>, <name> are skipped (inaudible/non-verbal)

SpaceAfter=No is derived from join="right" attribute on the token.

Structure mapping:
  <TEI xml:id="..."> → # newdoc id
  <u xml:id="...u#"> → # newpar id  (utterance ≈ paragraph)
  <seg xml:id="...s#"> → # sent_id

Usage:
    python scripts/gos/gos_tei_to_conllu.py \
        --tei_dir data/datasets/GOS/Gos.TEI \
        --output  data/datasets/GOS/gos.conllu
"""

import argparse
import logging
import re
import sys
from pathlib import Path

from conllu import TokenList
from lxml import etree

TEI_NS  = 'http://www.tei-c.org/ns/1.0'
XML_NS  = 'http://www.w3.org/XML/1998/namespace'
W       = f'{{{TEI_NS}}}w'
PC      = f'{{{TEI_NS}}}pc'
DEL     = f'{{{TEI_NS}}}del'
SEG     = f'{{{TEI_NS}}}seg'
U_TAG   = f'{{{TEI_NS}}}u'
BODY    = f'{{{TEI_NS}}}body'
XML_ID  = f'{{{XML_NS}}}id'

# Tags inside <seg> that contain speakable tokens
TOKEN_TAGS = {W, PC}
# Tags to recurse into for tokens (e.g. <del> contains <w>)
RECURSE_TAGS = {DEL}
# Tags to skip entirely
SKIP_TAGS = {
    f'{{{TEI_NS}}}gap',
    f'{{{TEI_NS}}}vocal',
    f'{{{TEI_NS}}}name',
    f'{{{TEI_NS}}}incident',
}

# Corpus-level metadata files to skip
SKIP_FILES = {'Gos2.1.xml', 'mte-msd.xml'}


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


def collect_tokens(seg, doc_id, seg_id, logger):
    """Collect (form, space_after) pairs from a <seg> element."""
    tokens = []

    def process(child):
        tag = child.tag
        if tag in TOKEN_TAGS:
            form = (child.text or '').strip()
            if not form:
                logger.warning(f'{doc_id}/{seg_id}: empty <{tag.split("}")[-1]}>, skipping')
                return
            space_after = child.get('join') != 'right'
            tokens.append((form, space_after))
        elif tag in RECURSE_TAGS:
            for sub in child:
                process(sub)
        elif tag in SKIP_TAGS:
            pass
        else:
            unknown = tag.split('}')[-1]
            logger.debug(f'{doc_id}/{seg_id}: skipping unexpected <{unknown}>')

    for child in seg:
        process(child)

    return tokens


def make_token(tok_id, form, space_after):
    return {
        'id':     tok_id,
        'form':   form,
        'lemma':  '_',
        'upos':   '_',
        'xpos':   '_',
        'feats':  None,
        'head':   0,
        'deprel': '_',
        'deps':   None,
        'misc':   None if space_after else {'SpaceAfter': 'No'},
    }


def convert_file(xml_path, logger):
    """Parse one GOS TEI XML file, return (token_lists, sent_count, tok_count)."""
    doc_id = xml_path.stem
    token_lists = []

    try:
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(str(xml_path), parser)
    except etree.XMLSyntaxError as e:
        logger.error(f'{doc_id}: XML parse error — {e}')
        return token_lists, 0, 0

    root = tree.getroot()
    # Use xml:id from root if present, else filename
    doc_id = root.get(XML_ID) or doc_id

    body = root.find(f'.//{BODY}')
    if body is None:
        logger.warning(f'{doc_id}: no <body> element, skipping')
        return token_lists, 0, 0

    sent_count = 0
    tok_count = 0
    first_sent_in_doc = True

    for utt in body.findall(f'.//{U_TAG}'):
        utt_id = utt.get(XML_ID) or f'{doc_id}.u?'
        first_sent_in_utt = True

        for seg in utt.findall(f'{SEG}'):
            seg_id = seg.get(XML_ID) or f'{doc_id}.s{sent_count + 1}'

            tokens = collect_tokens(seg, doc_id, seg_id, logger)

            if not tokens:
                logger.warning(f'{doc_id}/{seg_id}: no tokens, skipping')
                continue

            # Reconstruct sentence text
            text_parts = []
            for j, (form, space_after) in enumerate(tokens):
                text_parts.append(form)
                if space_after and j < len(tokens) - 1:
                    text_parts.append(' ')
            text = ''.join(text_parts)

            metadata = {}
            if first_sent_in_doc:
                metadata['newdoc id'] = doc_id
                first_sent_in_doc = False
            if first_sent_in_utt:
                metadata['newpar id'] = utt_id
                first_sent_in_utt = False
            metadata['sent_id'] = seg_id
            metadata['text'] = text

            token_data = [make_token(i, form, sa) for i, (form, sa) in enumerate(tokens, 1)]
            token_lists.append(TokenList(token_data, metadata=metadata))

            tok_count += len(tokens)
            sent_count += 1

    return token_lists, sent_count, tok_count


def main():
    parser = argparse.ArgumentParser(description='Convert GOS TEI to CoNLL-U')
    parser.add_argument('--tei_dir', default='data/datasets/GOS/Gos.TEI',
                        help='Root directory of GOS TEI XML files')
    parser.add_argument('--output', default='data/datasets/GOS/gos.conllu',
                        help='Output CoNLL-U file')
    parser.add_argument('--log', default=None,
                        help='Optional log file path')
    parser.add_argument('--limit', type=int, default=None,
                        help='Process only first N files (for testing)')
    args = parser.parse_args()

    logger = setup_logging(args.log)

    tei_dir = Path(args.tei_dir)
    manifest = tei_dir / 'Gos2.1.xml'
    if manifest.exists():
        xml_files = []
        with open(manifest) as f:
            for line in f:
                m = re.search(r'xi:include[^>]+href="([^"]+)"', line)
                if m:
                    xml_files.append((tei_dir / m.group(1)).resolve())
        xml_files = sorted(xml_files)
        logger.info(f'Loaded {len(xml_files)} files from manifest {manifest.name}')
    else:
        logger.warning(f'Gos2.1.xml not found, falling back to rglob')
        xml_files = sorted(f for f in tei_dir.rglob('*.xml')
                           if f.name not in SKIP_FILES)

    if not xml_files:
        logger.error(f'No .xml files found under {tei_dir}')
        sys.exit(1)

    if args.limit:
        xml_files = xml_files[:args.limit]
        logger.info(f'Processing first {args.limit} files (--limit)')

    logger.info(f'Found {len(xml_files)} TEI files under {tei_dir}')

    total_docs = 0
    total_sents = 0
    total_toks = 0

    with open(args.output, 'w', encoding='utf-8') as out:
        for i, xml_path in enumerate(xml_files):
            token_lists, sents, toks = convert_file(xml_path, logger)
            if not token_lists:
                continue

            for tl in token_lists:
                out.write(tl.serialize())

            total_docs += 1
            total_sents += sents
            total_toks += toks

            if (i + 1) % 100 == 0:
                logger.info(
                    f'Progress: {i + 1}/{len(xml_files)} files, '
                    f'{total_sents} sentences, {total_toks} tokens'
                )

    logger.info(
        f'Done. {total_docs} docs, {total_sents} sentences, {total_toks} tokens '
        f'→ {args.output}'
    )


if __name__ == '__main__':
    main()
