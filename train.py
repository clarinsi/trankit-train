import argparse

import trankit
import time
from trankit.utils.mwt_lemma_utils.seq2seq_utils import VOCAB_PREFIX, SOS, EOS


trankit.utils.mwt_lemma_utils.seq2seq_vocabs.EMPTY = SOS
trankit.utils.mwt_lemma_utils.seq2seq_vocabs.ROOT = EOS
trankit.utils.mwt_lemma_utils.seq2seq_vocabs.VOCAB_PREFIX = VOCAB_PREFIX

def read_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
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
    if args.tokenize:
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
#                'batch_size': 4
            }
        )

        # start training
        trainer.train()

    if args.mwt:
        # initialize a trainer for the task
        trainer = trankit.TPipeline(
            training_config={
                'category': args.category,  # pipeline category
                'task': 'mwt',  # task name
                'save_dir': args.save_dir,  # directory for saving trained model
                'train_conllu_fpath': args.train_conllu_fpath,  # annotations file in CONLLU format  for training
                'dev_conllu_fpath': args.dev_conllu_fpath,  # annotations file in CONLLU format for development
                'embedding': args.embedding
            }
        )

        # start training
        trainer.train()

    if args.posdep:
        # initialize a trainer for the task
        trainer = trankit.TPipeline(
            training_config={
                'category': args.category, # pipeline category
                'task': 'posdep', # task name
                'save_dir': args.save_dir, # directory for saving trained model
                'train_conllu_fpath': args.train_conllu_fpath, # annotations file in CONLLU format  for training
                'dev_conllu_fpath': args.dev_conllu_fpath, # annotations file in CONLLU format for development
                'embedding': args.embedding
            }
        )
        # start training
        trainer.train()

    if args.lemmatize:
        # initialize a trainer for the task
        trainer = trankit.TPipeline(
            training_config={
            'category': args.category, # pipeline category
            'task': 'lemmatize', # task name
            'save_dir': args.save_dir, # directory for saving trained model
            'train_conllu_fpath': args.train_conllu_fpath, # annotations file in CONLLU format  for training
            'dev_conllu_fpath': args.dev_conllu_fpath, # annotations file in CONLLU format for development
            'embeddings': args.embedding
            }
        )
        # start training
        trainer.train()


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Total:")
    print("--- %s seconds ---" % (time.time() - start_time))
