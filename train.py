import argparse
import logging
import time

# Fix dead HuggingFace CDN URLs in adapter_transformers 1.0.1 bundled with trankit 1.1.1
def _fixed_hf_bucket_url(identifier, filename, use_cdn=False, mirror=None):
    return f"https://huggingface.co/{identifier}/resolve/main/{filename}"

import trankit.adapter_transformers.modeling_utils as _mu
import trankit.adapter_transformers.configuration_utils as _cu
import trankit.adapter_transformers.tokenization_utils as _tu
_mu.hf_bucket_url = _fixed_hf_bucket_url
_cu.hf_bucket_url = _fixed_hf_bucket_url
_tu.hf_bucket_url = _fixed_hf_bucket_url

import trankit
from trankit.utils.mwt_lemma_utils.seq2seq_utils import VOCAB_PREFIX, SOS, EOS


trankit.utils.mwt_lemma_utils.seq2seq_vocabs.EMPTY = SOS
trankit.utils.mwt_lemma_utils.seq2seq_vocabs.ROOT = EOS
trankit.utils.mwt_lemma_utils.seq2seq_vocabs.VOCAB_PREFIX = VOCAB_PREFIX

# Fix trankit 1.1.1 bug: CoNLL.conll_as_string strips trailing '_' fields, producing
# fewer than 10 columns, which the UD evaluator rejects. Pad each row to 10 fields.
from trankit.utils.conll import CoNLL as _CoNLL

@staticmethod
def _fixed_conll_as_string(doc):
    _FIELD_NUM = 10
    return_string = ""
    for sent in doc:
        for ln in sent:
            while len(ln) < _FIELD_NUM:
                ln.append('_')
            return_string += ("\t".join(ln[:_FIELD_NUM]) + "\n")
        return_string += "\n"
    return return_string

_CoNLL.conll_as_string = _fixed_conll_as_string

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def read_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--category", type=str, default='customized', help="pipeline category")
    parser.add_argument("--posdep", action="store_true", default=False, help="task name")
    parser.add_argument("--mwt", action="store_true", default=False, help="task name")
    parser.add_argument("--lemmatize", action="store_true", default=False, help="task name")
    parser.add_argument("--tokenize", action="store_true", default=False, help="task name")
    parser.add_argument("--embedding", type=str, default='xlm-roberta-base', help="task name")
    parser.add_argument("--save_dir", type=str, default='data/save_dir', help="directory for saving trained model")
    parser.add_argument("--train_conllu_fpath", type=str, default='data/ssj/sl_ssj-ud-train.conllu', help="annotations file in CONLLU format  for training")
    parser.add_argument("--dev_conllu_fpath", type=str, default='data/ssj/sl_ssj-ud-dev.conllu', help="annotations file in CONLLU format for development")
    parser.add_argument("--train_txt_fpath", type=str, default='data/ssj/train.txt', help="raw text file")
    parser.add_argument("--dev_txt_fpath", type=str, default='data/ssj/dev.txt', help="raw text file")
    return parser.parse_args()


def main():
    args = read_args()
    start_time = time.time()
    tasks = [t for t in ['tokenize', 'posdep', 'lemmatize'] if getattr(args, t)]
    logger.info(f"Starting training: {', '.join(tasks) or 'none'}")
    logger.info(f"  embedding: {args.embedding}")
    logger.info(f"  save_dir:  {args.save_dir}")

    if args.tokenize:
        logger.info("--- tokenize: initializing ---")
        # initialize a trainer for the task
        trainer = trankit.TPipeline(
            training_config={
                'category': args.category,  # pipeline category
                'task': 'tokenize',  # task name
                'save_dir': args.save_dir,  # directory for saving trained model
                'train_txt_fpath': args.train_txt_fpath,  # raw text file
                'train_conllu_fpath': args.train_conllu_fpath,  # annotations file in CONLLU format for training
                'dev_txt_fpath': args.dev_txt_fpath,  # raw text file
                'dev_conllu_fpath': args.dev_conllu_fpath,  # annotations file in CONLLU format for development
                'embedding': args.embedding
            }
        )

        logger.info("--- tokenize: training ---")
        trainer.train()
        logger.info("--- tokenize: done ---")

    if args.posdep:
        logger.info("--- posdep: initializing ---")
        # initialize a trainer for the task
        trainer = trankit.TPipeline(
            training_config={
                'category': args.category, # pipeline category
                'task': 'posdep', # task name
                'save_dir': args.save_dir, # directory for saving trained model
                'train_conllu_fpath': args.train_conllu_fpath, # annotations file in CONLLU format  for training
                'dev_conllu_fpath': args.dev_conllu_fpath, # annotations file in CONLLU format for development
                'embedding': args.embedding,
                'batch_size': 12
            }
        )
        logger.info("--- posdep: training ---")
        trainer.train()
        logger.info("--- posdep: done ---")

    if args.lemmatize:
        logger.info("--- lemmatize: initializing ---")
        # initialize a trainer for the task
        trainer = trankit.TPipeline(
            training_config={
            'category': args.category, # pipeline category
            'task': 'lemmatize', # task name
            'save_dir': args.save_dir, # directory for saving trained model
            'train_conllu_fpath': args.train_conllu_fpath, # annotations file in CONLLU format  for training
            'dev_conllu_fpath': args.dev_conllu_fpath, # annotations file in CONLLU format for development
            'embedding': args.embedding
            }
        )
        logger.info("--- lemmatize: training ---")
        trainer.train()
        logger.info("--- lemmatize: done ---")


if __name__ == "__main__":
    start_time = time.time()
    main()
    logger.info(f"All done. Total: {time.time() - start_time:.0f}s")
