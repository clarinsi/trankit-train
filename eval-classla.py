import argparse

import trankit
import time

from trankit import Pipeline, trankit2conllu
from trankit.utils import CoNLL, get_ud_score, get_ud_performance_table
import classla


def read_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--category", type=str, default='customized', help="pipeline category")
    parser.add_argument("--posdep", action="store_true", default=False, help="task name")
    parser.add_argument("--lemmatize", action="store_true", default=False, help="task name")
    parser.add_argument("--tokenize", action="store_true", default=False, help="task name")
    parser.add_argument("--embedding", type=str, default='xlm-roberta-base', help="task name")
    parser.add_argument("--save_dir", type=str, default='data/save_dir', help="directory for saving trained model")
    parser.add_argument("--type", type=str, default='standard', help="directory for saving trained model")
    parser.add_argument("--test_conllu_fpath", type=str, default='data/ssj/sl_ssj-ud-test.conllu', help="annotations file in CONLLU format for testing")
    parser.add_argument("--test_txt_fpath", type=str, default='data/ssj/test.txt', help="raw text file")
    parser.add_argument("--pred_conllu_fpath", type=str, default='data/ssj/predictions.conllu', help="predicted conllu")
    parser.add_argument("--classla_spoken", action="store_true", default=False, help="task name")
    return parser.parse_args()


def main():
    args = read_args()

    # VERIFY FOR MISSING PARTS
    # trankit.verify_customized_pipeline(
    #     category=args.category,  # pipeline category
    #     save_dir=args.save_dir,  # directory used for saving models in previous steps
    #     embedding_name='xlm-roberta-large'
    #     # embedding version that we use for training our customized pipeline, by default, it is `xlm-roberta-base`
    # )

    # DOWNLOAD MISSING PARTS WHEN NECESSARY
    # trankit.download_missing_files(
    #     category=args.category,  # pipeline category
    #     save_dir=args.save_dir,  # directory used for saving models in previous steps
    #     embedding_name='xlm-roberta-large',
    #     language='slovenian'
    #     # embedding version that we use for training our customized pipeline, by default, it is `xlm-roberta-base`
    # )



    infile = open(args.test_conllu_fpath)
    gold_tokens = CoNLL.load_conll(infile)
    infile.close()
    gold_tokens = [[tok[1] for tok in sent] for sent in gold_tokens]
    classla.download('sl', type=args.type)
    if args.classla_spoken:
        nlp = classla.Pipeline('sl', pos_use_lexicon=True,
                               pos_model_path='/home/luka/Development/CJVT/trankit-train/data/classla-spoken-models/pos/baseline+ssj500+janes',
                               lemma_model_path='/home/luka/Development/CJVT/trankit-train/data/classla-spoken-models/lemma/baseline+ssj500_lemmatizer.pt',
                               depparse_model_path='/home/luka/Development/CJVT/trankit-train/data/classla-spoken-models/depparse/baseline+ssj',
                               pos_pretrain_path='/home/luka/Development/CJVT/trankit-train/data/classla-spoken-models/sl_ssj.pretrain.pt',
                               depparse_pretrain_path='/home/luka/Development/CJVT/trankit-train/data/classla-spoken-models/sl_ssj.pretrain.pt',
                               type=args.type, tokenize_pretokenized=True)
    else:
        nlp = classla.Pipeline('sl', pos_use_lexicon=True, type=args.type, tokenize_pretokenized=True)

    doc = nlp(gold_tokens)

    with open(args.pred_conllu_fpath, 'w') as wf:
        wf.write(doc.to_conll())
    score = get_ud_score(args.test_conllu_fpath, args.pred_conllu_fpath)
    print(get_ud_performance_table(score))


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Total:")
    print("--- %s seconds ---" % (time.time() - start_time))
