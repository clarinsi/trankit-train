import argparse
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

from trankit import Pipeline, trankit2conllu
from trankit.utils import CoNLL, get_ud_score, get_ud_performance_table
import trankit


def read_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--category", type=str, default='customized', help="pipeline category")
    parser.add_argument("--raw_input", action="store_true", default=False, help="signals that we don't have input in conllu format")
    parser.add_argument("--embedding", type=str, default='xlm-roberta-base', help="task name")
    parser.add_argument("--save_dir", type=str, default='data/save_dir', help="directory for saving trained model")
    parser.add_argument("--test_conllu_fpath", type=str, default='data/ssj/sl_ssj-ud-test.conllu', help="annotations file in CONLLU format for testing")
    parser.add_argument("--test_txt_fpath", type=str, default='data/ssj/test.txt', help="raw text file")
    parser.add_argument("--pred_conllu_fpath", type=str, default='data/ssj/predictions.conllu', help="predicted conllu")
    return parser.parse_args()


def _fix_multiple_roots(conllu_str):
    """Fix sentences where Trankit predicted multiple root tokens (HEAD=0).
    Keeps the first root; re-attaches subsequent roots to the first root token
    with deprel='dep'."""
    fixed_lines = []
    first_root_id = None
    for line in conllu_str.splitlines(keepends=True):
        if line.strip() == '':
            first_root_id = None
            fixed_lines.append(line)
            continue
        if line.startswith('#'):
            fixed_lines.append(line)
            continue
        parts = line.rstrip('\n').split('\t')
        if len(parts) >= 8 and parts[6] == '0':
            if first_root_id is None:
                first_root_id = parts[0]
            else:
                parts[6] = first_root_id
                parts[7] = 'dep'
                line = '\t'.join(parts) + '\n'
        fixed_lines.append(line)
    return ''.join(fixed_lines)


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
    #     language='slovenian-sst'
    #     # embedding version that we use for training our customized pipeline, by default, it is `xlm-roberta-base`
    # )
    if args.raw_input:
        with open(args.test_txt_fpath, 'r', encoding='utf-8') as rf:
            gold_tokens = rf.read()
    else:
        infile = open(args.test_conllu_fpath)
        gold_tokens = CoNLL.load_conll(infile, ignore_gapping=False)
        infile.close()
        gold_tokens = [[tok[1] for tok in sent] for sent in gold_tokens]
    p = Pipeline(lang=args.category, cache_dir=args.save_dir, embedding=args.embedding)
    execution_time_start = time.time()
    all_trankit = p(gold_tokens)
    print("Execution time:")
    print("--- %s seconds ---" % (time.time() - execution_time_start))
    all_conllu = trankit2conllu(all_trankit)
    all_conllu = _fix_multiple_roots(all_conllu)
    with open(args.pred_conllu_fpath, 'w') as wf:
        wf.write(all_conllu)

    if not args.raw_input:
        score = get_ud_score(args.test_conllu_fpath, args.pred_conllu_fpath)
        print(get_ud_performance_table(score))


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("Total:")
    print("--- %s seconds ---" % (time.time() - start_time))
