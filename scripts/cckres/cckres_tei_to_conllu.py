#!/usr/bin/env python3
"""Convert ccKres TEI XML files to CoNLL-U for Trankit annotation.

Reads cckresV1_0/*.xml, outputs a single CoNLL-U file.
- FORM tokens with SpaceAfter=No in MISC derived from <S/> elements
- Sentence/paragraph/document IDs from TEI attributes
- All other CoNLL-U columns set to _ (filled by Trankit later)

Usage:
    python scripts/cckres_tei_to_conllu.py \
        --tei_dir data/datasets/ccKres/cckresV1_0 \
        --output  data/datasets/ccKres/cckres.conllu
"""

import argparse
import logging
import sys
from pathlib import Path

from conllu import TokenList
from lxml import etree

TEI_NS  = 'http://www.tei-c.org/ns/1.0'
XML_NS  = 'http://www.w3.org/XML/1998/namespace'
W       = f'{{{TEI_NS}}}w'
C       = f'{{{TEI_NS}}}c'
S_TAG   = f'{{{TEI_NS}}}S'
P_TAG   = f'{{{TEI_NS}}}p'
SENT    = f'{{{TEI_NS}}}s'
BODY    = f'{{{TEI_NS}}}body'
XML_ID  = f'{{{XML_NS}}}id'

KNOWN_TAGS = {W, C, S_TAG}


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
        'misc':   {'SpaceAfter': 'No'} if space_after is False else None,
    }


def convert_file(xml_path, logger):
    """Parse one TEI XML file and return (token_lists, sent_count, tok_count)."""
    doc_id = xml_path.stem
    token_lists = []

    try:
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(str(xml_path), parser)
    except etree.XMLSyntaxError as e:
        logger.error(f'{doc_id}: XML parse error — {e}')
        return token_lists, 0, 0

    root = tree.getroot()
    body = root.find(f'.//{BODY}')
    if body is None:
        logger.warning(f'{doc_id}: no <body> element, skipping')
        return token_lists, 0, 0

    sent_count = 0
    tok_count = 0
    first_sent_in_doc = True

    for para_i, para in enumerate(body.findall(f'{P_TAG}')):
        para_id = para.get(XML_ID) or f'{doc_id}.p{para_i + 1}'
        first_sent_in_para = True

        for sent in para.findall(f'{SENT}'):
            children = list(sent)

            unknown = {ch.tag for ch in children if ch.tag not in KNOWN_TAGS}
            if unknown:
                logger.warning(
                    f'{doc_id}/{para_id}: unexpected tags in <s>: '
                    f'{[t.split("}")[-1] for t in unknown]}'
                )

            # Build tokens with space_after:
            #   True  = <S/> follows → space present → misc=None
            #   False = <w>/<c> follows directly → SpaceAfter=No
            #   None  = sentence-final → misc=None
            tokens = []
            relevant = [(ch, ch.tag) for ch in children if ch.tag in KNOWN_TAGS]

            for idx, (child, tag) in enumerate(relevant):
                if tag == S_TAG:
                    continue

                form = (child.text or '').strip()
                if not form:
                    logger.warning(
                        f'{doc_id}/{para_id}: empty <{tag.split("}")[-1]}>, skipping'
                    )
                    continue

                next_tag = relevant[idx + 1][1] if idx + 1 < len(relevant) else None
                if next_tag == S_TAG:
                    space_after = True
                elif next_tag in (W, C):
                    space_after = False
                else:
                    space_after = None

                tokens.append((form, space_after))

            if not tokens:
                logger.warning(f'{doc_id}/{para_id}: empty sentence, skipping')
                continue

            # Reconstruct sentence text
            text_parts = []
            for j, (form, space_after) in enumerate(tokens):
                text_parts.append(form)
                if space_after is True and j < len(tokens) - 1:
                    text_parts.append(' ')
            text = ''.join(text_parts)

            sent_id = f'{para_id}.s{sent_count + 1}'

            metadata = {}
            if first_sent_in_doc:
                metadata['newdoc id'] = doc_id
                first_sent_in_doc = False
            if first_sent_in_para:
                metadata['newpar id'] = para_id
                first_sent_in_para = False
            metadata['sent_id'] = sent_id
            metadata['text'] = text

            token_data = [make_token(i, form, sa) for i, (form, sa) in enumerate(tokens, 1)]
            token_lists.append(TokenList(token_data, metadata=metadata))

            tok_count += len(tokens)
            sent_count += 1

    return token_lists, sent_count, tok_count


def main():
    parser = argparse.ArgumentParser(description='Convert ccKres TEI to CoNLL-U')
    parser.add_argument('--tei_dir', default='data/datasets/ccKres/cckresV1_0',
                        help='Directory of TEI XML files')
    parser.add_argument('--output', default='data/datasets/ccKres/cckres.conllu',
                        help='Output CoNLL-U file')
    parser.add_argument('--log', default=None,
                        help='Optional log file path')
    parser.add_argument('--limit', type=int, default=None,
                        help='Process only first N files (for testing)')
    args = parser.parse_args()

    logger = setup_logging(args.log)

    tei_dir = Path(args.tei_dir)
    xml_files = sorted(tei_dir.glob('*.xml'))
    if not xml_files:
        logger.error(f'No .xml files found in {tei_dir}')
        sys.exit(1)

    if args.limit:
        xml_files = xml_files[:args.limit]
        logger.info(f'Processing first {args.limit} files (--limit)')

    logger.info(f'Found {len(xml_files)} TEI files in {tei_dir}')

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

            if (i + 1) % 500 == 0:
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
